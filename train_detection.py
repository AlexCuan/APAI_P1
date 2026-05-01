import os
import time
import json
import numpy as np
import cv2
import tifffile
import matplotlib.pyplot as plt
from collections import defaultdict
from natsort import natsorted
from tqdm import tqdm
from shapely.wkt import loads
from shapely.geometry import Polygon

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.models as models
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import warnings

warnings.filterwarnings("ignore")  # Para evitar los warnings molestos de PyTorch

# ==============================================================================
# 1. CLASES DE DATASET (INTEGRADAS)
# ==============================================================================

DAMAGE_CLASSES = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}
TASKS = {"classification", "detection", "segmentation"}
IMAGE_SIZE = 1024


class xBDDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: list = None,
        task: str = "classification",
        transform=None,
        patch_size: int = 64,
        stride: int = None,
        stats: dict = None,
        max_size: int = 0,
    ):
        if split is None:
            split = ["train"]
        self.data_dir = data_dir
        self.split = split
        self.task = task
        self.transform = transform
        self.max_size = max_size
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size

        self.image_pre_files, self.image_post_files = [], []
        self.label_pre_files, self.label_post_files = [], []

        for s in split:
            split_dir = os.path.join(data_dir, s)
            for d in sorted(os.listdir(split_dir)):
                img_dir = os.path.join(split_dir, d, "images")
                lbl_dir = os.path.join(split_dir, d, "labels")
                imgs = os.listdir(img_dir)
                lbls = os.listdir(lbl_dir)
                self.image_pre_files.append(
                    natsorted([os.path.join(img_dir, f) for f in imgs if "pre" in f])
                )
                self.image_post_files.append(
                    natsorted([os.path.join(img_dir, f) for f in imgs if "post" in f])
                )
                self.label_pre_files.append(
                    natsorted([os.path.join(lbl_dir, f) for f in lbls if "pre" in f])
                )
                self.label_post_files.append(
                    natsorted([os.path.join(lbl_dir, f) for f in lbls if "post" in f])
                )

        self.image_pre_files = natsorted(np.hstack(self.image_pre_files))
        self.image_post_files = natsorted(np.hstack(self.image_post_files))
        self.label_pre_files = natsorted(np.hstack(self.label_pre_files))
        self.label_post_files = natsorted(np.hstack(self.label_post_files))

        if self.max_size > 0:
            # Si estamos en test, semilla aleatoria. Si estamos en train, semilla fija.
            if "test" in self.split:
                rng = np.random.RandomState()  # Aleatorio de verdad
            else:
                rng = np.random.RandomState(seed=42)  # Fijo para reproducibilidad

            idx = rng.permutation(range(len(self.image_pre_files)))

            self.image_post_files = np.array(self.image_post_files)[
                idx[0 : self.max_size]
            ]
            self.label_post_files = np.array(self.label_post_files)[
                idx[0 : self.max_size]
            ]

        self.samples = []
        self._build_samples()

    @staticmethod
    def _load_json(path: str) -> dict:
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _read_image(path: str) -> np.ndarray:
        img = tifffile.imread(path)
        if img.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        return img

    def _parse_buildings(self, data_post: dict) -> list:
        buildings = []
        for feat in data_post.get("features", {}).get("xy", []):
            props = feat.get("properties", {})
            if props.get("feature_type") != "building":
                continue
            wkt = feat.get("wkt", "")
            dmg = props.get("subtype", "no-damage")
            if dmg not in DAMAGE_CLASSES and dmg not in [-1, "un-classified"]:
                continue
            try:
                geom = loads(wkt)
                if not (isinstance(geom, Polygon) and geom.is_valid):
                    continue
                coords = np.array(geom.exterior.coords, dtype=np.float32)
                x1, y1 = coords.min(axis=0)
                x2, y2 = coords.max(axis=0)
                buildings.append(
                    {
                        "wkt": wkt,
                        "label": DAMAGE_CLASSES[dmg] if dmg in DAMAGE_CLASSES else -1,
                        "bbox_full": [float(x1), float(y1), float(x2), float(y2)],
                    }
                )
            except Exception:
                continue
        return buildings

    @staticmethod
    def _centred_window(cx: float, cy: float, ps: int):
        half = ps // 2
        x1 = int(np.clip(round(cx) - half, 0, IMAGE_SIZE - ps))
        y1 = int(np.clip(round(cy) - half, 0, IMAGE_SIZE - ps))
        return x1, y1, x1 + ps, y1 + ps

    @staticmethod
    def _dedup_windows(origins: list, stride: int):
        if stride == 0 or not origins:
            return origins
        accepted = []
        for ox, oy in origins:
            too_close = any(
                abs(ox - ax) <= stride and abs(oy - ay) <= stride for ax, ay in accepted
            )
            if not too_close:
                accepted.append((ox, oy))
        return accepted

    def _windows_for_image(self, buildings: list, ps: int):
        candidates = []
        for b in buildings:
            bx1, by1, bx2, by2 = b["bbox_full"]
            cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            x1, y1, _, _ = self._centred_window(cx, cy, ps)
            candidates.append((x1, y1))
        candidates.sort(key=lambda o: (o[1], o[0]))
        unique_origins = self._dedup_windows(candidates, self.stride)
        return [(x1, y1, x1 + ps, y1 + ps) for x1, y1 in unique_origins]

    def _build_samples(self):
        ps = min(self.patch_size, IMAGE_SIZE)
        for img_post, lbl_post in zip(self.image_post_files, self.label_post_files):
            data_post = self._load_json(lbl_post)
            buildings = self._parse_buildings(data_post)

            if not buildings:
                continue
            windows = self._windows_for_image(buildings, ps)

            for wx1, wy1, wx2, wy2 in windows:
                local_buildings = []
                for b in buildings:
                    bx1, by1, bx2, by2 = b["bbox_full"]
                    cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                    if wx1 <= cx < wx2 and wy1 <= cy < wy2:
                        local_buildings.append(
                            {
                                **b,
                                "bbox_local": [
                                    float(np.clip(bx1 - wx1, 0, ps)),
                                    float(np.clip(by1 - wy1, 0, ps)),
                                    float(np.clip(bx2 - wx1, 0, ps)),
                                    float(np.clip(by2 - wy1, 0, ps)),
                                ],
                            }
                        )
                self.samples.append(
                    {
                        "img_post": img_post,
                        "lbl_post": lbl_post,
                        "buildings": local_buildings,
                        "window": (wx1, wy1, wx2, wy2),
                    }
                )

    def _crop_window(self, image: np.ndarray, window: tuple) -> np.ndarray:
        x1, y1, x2, y2 = window
        patch = image[y1:y2, x1:x2]
        if patch.shape[:2] != (self.patch_size, self.patch_size):
            patch = cv2.resize(patch, (self.patch_size, self.patch_size))
        return patch

    def _bboxes_for_sample(self, sample: dict):
        boxes, labels = [], []
        for b in sample["buildings"]:
            if b["label"] < 0:
                continue
            boxes.append(b.get("bbox_local", b["bbox_full"]))
            labels.append(b["label"])
        if boxes:
            boxes = np.array(np.round(boxes), dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        return boxes, labels

    def _normalise(self, patch: np.ndarray) -> np.ndarray:
        patch = patch.astype(np.float32) / 255.0
        return patch

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        img_post = self._read_image(sample["img_post"])
        patch_post = self._crop_window(img_post, sample["window"])
        patch_post = torch.from_numpy(
            np.transpose(self._normalise(patch_post), (2, 0, 1))
        )

        base = {
            "patch_post_path": sample["img_post"],
            "patch_post": patch_post,
            "idx": idx,
        }

        if self.task == "detection":
            boxes, labels = self._bboxes_for_sample(sample)
            base["boxes"] = torch.from_numpy(boxes)
            base["labels"] = torch.from_numpy(labels)

        return base


class xBDDetectionDataset(Dataset):
    def __init__(self, data_dir, split, patch_size=512, max_size=0):
        self._base = xBDDataset(
            data_dir=data_dir,
            split=split,
            task="detection",
            patch_size=patch_size,
            max_size=max_size,
        )

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        sample = self._base[idx]
        image = sample["patch_post"].float()
        boxes = sample["boxes"]
        labels = sample["labels"]

        if boxes.size(0) > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            valid_idx = area > 0
            boxes = boxes[valid_idx]
            labels = labels[valid_idx]
            area = area[valid_idx]
        else:
            area = torch.tensor([0.0])

        target = {"boxes": boxes, "labels": labels + 1, "area": area}
        return image, target, sample["patch_post_path"]


def collate_fn(batch):
    return tuple(zip(*batch))


# ==============================================================================
# 2. CONSTRUCCIÓN DEL MODELO (VGG11 BACKBONE)
# ==============================================================================


def build_vgg_faster_rcnn(weights_path=None, num_classes=5):
    print("\nConstruyendo Backbone VGG11_bn para Detección...")
    vgg = models.vgg11_bn(weights=None)

    vgg.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    vgg.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(128, 4),
    )

    if weights_path and os.path.exists(weights_path):
        try:
            # Aquí cargamos los pesos del modelo VGG11_bn (best_ft_net.pth)
            vgg.load_state_dict(
                torch.load(weights_path, map_location="cpu", weights_only=True)
            )
            print(f"¡ÉXITO! Pesos cargados desde '{weights_path}'.")
        except Exception as e:
            print(f"Aviso: No se pudieron cargar los pesos: {e}")
    else:
        print(f"Aviso: El archivo '{weights_path}' no existe. Entrenando desde cero.")

    backbone = vgg.features
    backbone.out_channels = 512

    anchor_generator = AnchorGenerator(
        sizes=((16, 32, 64, 128, 256),), aspect_ratios=((0.5, 1.0, 2.0),)
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0"], output_size=7, sampling_ratio=2
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )

    return model


# ==============================================================================
# 3. BUCLE DE ENTRENAMIENTO OPTIMIZADO
# ==============================================================================


def train_detection():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"==========================================")
    print(f"Iniciando Entrenamiento Optimizado en {device}")
    print(f"==========================================")

    # Parches de 512x512 para ver contexto
    train_dataset = xBDDetectionDataset(
        data_dir="./xBD_UC3M", split=["train"], patch_size=512
    )

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=2, collate_fn=collate_fn
    )

    model = build_vgg_faster_rcnn("best_ft_net.pth", num_classes=5)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=1e-4, weight_decay=1e-3)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    # Usamos AMP (Automatic Mixed Precision) para ahorrar otro ~40% de VRAM
    scaler = torch.amp.GradScaler("cuda")

    accumulation_steps = (
        4  # Emula un batch_size=4 actualizando los pesos cada 4 imágenes
    )
    num_epochs = 15
    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()  # Inicializamos gradientes al empezar el epoch

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for i, (images, targets, paths) in enumerate(progress_bar):
            images = list(image.to(device) for image in images)

            valid_targets, valid_images = [], []
            for img, t in zip(images, targets):
                if len(t["boxes"]) > 0:
                    valid_targets.append({k: v.to(device) for k, v in t.items()})
                    valid_images.append(img)

            if len(valid_images) == 0:
                continue

            # Forward pass con Mixed Precision
            with torch.amp.autocast("cuda"):
                loss_dict = model(valid_images, valid_targets)
                losses = sum(loss for loss in loss_dict.values())
                # Dividimos la loss entre accumulation_steps para normalizar los gradientes
                losses = losses / accumulation_steps

            # Backward pass escalado
            scaler.scale(losses).backward()

            # Acumulación: actualizamos solo cada 'accumulation_steps'
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += losses.item() * accumulation_steps
            progress_bar.set_postfix(loss=f"{(losses.item() * accumulation_steps):.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"\n=> Epoch {epoch + 1} Finalizada - Avg Loss: {avg_loss:.4f}\n")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_detector_vgg.pth")
            print(
                ">>> [GUARDADO] Nuevo mejor modelo guardado como 'best_detector_vgg.pth'\n"
            )

        torch.cuda.empty_cache()  # Liberamos basura residual de VRAM al final del epoch


# ==============================================================================
# 4. INFERENCIA Y TESTEO
# ==============================================================================


def test_and_visualize():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"==========================================")
    print(
        f"Iniciando Fase de Test y Visualización Doble (Predicciones vs Ground Truth)"
    )
    print(f"==========================================")

    model = build_vgg_faster_rcnn(None, num_classes=5)
    if os.path.exists("best_detector_vgg.pth"):
        model.load_state_dict(
            torch.load("best_detector_vgg.pth", map_location=device, weights_only=True)
        )
        print("Pesos del detector cargados correctamente.")
    else:
        print("ADVERTENCIA: No se encontró 'best_detector_vgg.pth'.")

    model.to(device)
    model.eval()

    test_dataset = xBDDetectionDataset(
        data_dir="./xBD_UC3M", split=["test"], patch_size=512, max_size=10
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn
    )

    # Clases abreviadas
    class_names = ["Fondo", "ND", "MinD", "MajD", "Dest"]
    colors = [(0, 0, 0), (0, 255, 0), (0, 255, 255), (255, 165, 0), (255, 0, 0)]

    with torch.no_grad():
        for images, targets, paths in test_loader:
            img_tensor = images[0].to(device)

            with torch.amp.autocast("cuda"):
                predictions = model([img_tensor])[0]

            # 1. Predicciones
            boxes = predictions["boxes"].cpu().numpy()
            labels = predictions["labels"].cpu().numpy()
            scores = predictions["scores"].cpu().numpy()

            # 2. Etiquetas Reales (Ground Truth)
            gt_boxes = targets[0]["boxes"].cpu().numpy()
            gt_labels = targets[0]["labels"].cpu().numpy()

            umbral = 0.80

            # --- PREPARAMOS DOS COPIAS DE LA IMAGEN ---
            img_base = img_tensor.cpu().permute(1, 2, 0).numpy().copy()
            img_base = (img_base * 255).astype(np.uint8)
            img_base = cv2.cvtColor(img_base, cv2.COLOR_RGB2BGR)

            img_cv_pred = img_base.copy()  # Para pintar lo que dice la red
            img_cv_gt = img_base.copy()  # Para pintar la realidad

            found_objects = 0
            correct_predictions = 0

            # ==========================================
            # PANEL 1: DIBUJAR PREDICCIONES
            # ==========================================
            for i in range(len(boxes)):
                if scores[i] >= umbral:
                    found_objects += 1
                    box = boxes[i].astype(int)
                    pred_label = labels[i]
                    score = scores[i]
                    color = colors[pred_label]

                    # Lógica de comprobación de Acierto (IoU)
                    is_correct = False
                    if len(gt_boxes) > 0:
                        box_t = torch.tensor(box).unsqueeze(0).float()
                        gt_boxes_t = torch.tensor(gt_boxes).float()

                        ious = torchvision.ops.box_iou(box_t, gt_boxes_t)[0]
                        max_iou, max_idx = ious.max(dim=0)

                        if max_iou >= 0.5:
                            gt_label = gt_labels[max_idx.item()]
                            if pred_label == gt_label:
                                is_correct = True

                    if is_correct:
                        correct_predictions += 1

                    # Dibujar Rectángulo principal
                    cv2.rectangle(
                        img_cv_pred, (box[0], box[1]), (box[2], box[3]), color, 2
                    )

                    # Escribir Etiqueta
                    text = f"{class_names[pred_label]} {score:.2f}"
                    cv2.putText(
                        img_cv_pred,
                        text,
                        (box[0], box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                    )

                    # Dibujar CRUZ o CHECK
                    cx = int((box[0] + box[2]) / 2)
                    cy = int((box[1] + box[3]) / 2)

                    if is_correct:
                        # Check verde (✔)
                        cv2.line(
                            img_cv_pred, (cx - 4, cy), (cx - 1, cy + 4), (0, 255, 0), 2
                        )
                        cv2.line(
                            img_cv_pred,
                            (cx - 1, cy + 4),
                            (cx + 5, cy - 4),
                            (0, 255, 0),
                            2,
                        )
                    else:
                        # Cruz roja (❌)
                        cv2.line(
                            img_cv_pred,
                            (cx - 4, cy - 4),
                            (cx + 4, cy + 4),
                            (0, 0, 255),
                            2,
                        )
                        cv2.line(
                            img_cv_pred,
                            (cx + 4, cy - 4),
                            (cx - 4, cy + 4),
                            (0, 0, 255),
                            2,
                        )

            # ==========================================
            # PANEL 2: DIBUJAR GROUND TRUTH (REALIDAD)
            # ==========================================
            for i in range(len(gt_boxes)):
                box = gt_boxes[i].astype(int)
                gt_label = gt_labels[i]
                color = colors[gt_label]

                # Dibujar Rectángulo
                cv2.rectangle(img_cv_gt, (box[0], box[1]), (box[2], box[3]), color, 2)

                # Escribir Etiqueta (solo nombre, sin score)
                text = f"{class_names[gt_label]}"
                cv2.putText(
                    img_cv_gt,
                    text,
                    (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )

            # --- VISUALIZACIÓN DOBLE (MATPLOTLIB) ---
            nombre_archivo = os.path.basename(paths[0])
            tipo_imagen = "POST" if "post" in nombre_archivo.lower() else "PRE"

            etiquetas_validas = [
                labels[i] for i in range(len(boxes)) if scores[i] >= umbral
            ]
            hay_danos = any(et > 1 for et in etiquetas_validas)
            estado_danos = "Daños" if hay_danos else "No Daños"

            print(
                f"Imagen: {nombre_archivo} -> Detectados: {found_objects} | Aciertos puros: {correct_predictions}"
            )

            # Convertir de BGR a RGB para Matplotlib
            img_rgb_pred = cv2.cvtColor(img_cv_pred, cv2.COLOR_BGR2RGB)
            img_rgb_gt = cv2.cvtColor(img_cv_gt, cv2.COLOR_BGR2RGB)

            # Crear figura con 2 filas y 1 columna
            fig, axes = plt.subplots(2, 1, figsize=(10, 16))

            # Arriba: Predicciones
            titulo_pred = f"PREDICCIONES ({tipo_imagen}) | Umbral {umbral} | {estado_danos} | Aciertos: {correct_predictions}/{found_objects}"
            axes[0].imshow(img_rgb_pred)
            axes[0].set_title(
                titulo_pred,
                fontsize=12,
                fontweight="bold",
                color="red" if hay_danos else "green",
            )
            axes[0].axis("off")

            # Abajo: Ground Truth
            titulo_gt = f"GROUND TRUTH"
            axes[1].imshow(img_rgb_gt)
            axes[1].set_title(titulo_gt, fontsize=12, fontweight="bold", color="blue")
            axes[1].axis("off")

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    print("Selecciona una opción:")
    print("1. Entrenar modelo de Detección")
    print("2. Testear y Visualizar resultados")

    opcion = input("Opción (1/2): ")
    if opcion == "1":
        train_detection()
    elif opcion == "2":
        test_and_visualize()
    else:
        print("Opción no válida.")
