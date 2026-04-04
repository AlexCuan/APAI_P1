"""
grid_search.py

Script automatizado de Grid Search para la arquitectura CustomNet.
Itera sobre múltiples configuraciones de hiperparámetros y arquitecturas.
Guarda los resultados, los mejores modelos y los logs en ./grid_training/
"""

import os
import time
import copy
import math
import random
import csv
import itertools
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard.writer import SummaryWriter
from PIL import Image
from natsort import natsorted
from tqdm import tqdm
from sklearn.metrics import f1_score

import warnings

warnings.filterwarnings("ignore")

NUM_EPOCHS = 20
BATCH_SIZE = 128

GRID_PARAMS = {
    "arch": ["3Block", "4Block"],  # Arquitecturas Custom
    "aug": ["Basic", "Advanced"],  # Data Augmentation
    "weights": ["None", "Sqrt"],  # Pesado de clases
    "loss": ["CE", "CE_Smooth0.1"],  # Función de pérdida
    "opt": ["SGD", "AdamW"],  # Optimizadores
    "sched": ["Step", "Cosine"],  # Schedulers
}

BASE_DIR = "./grid_training"
MODELS_DIR = os.path.join(BASE_DIR, "models")
BEST_DIR = os.path.join(BASE_DIR, "best")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
TB_DIR = os.path.join(BASE_DIR, "tb")

for d in [BASE_DIR, MODELS_DIR, BEST_DIR, LOGS_DIR, TB_DIR]:
    os.makedirs(d, exist_ok=True)

log_file = os.path.join(LOGS_DIR, "grid_search_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)

csv_file = os.path.join(LOGS_DIR, "results.csv")
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "Run_ID",
            "Architecture",
            "Augmentation",
            "ClassWeights",
            "Loss",
            "Optimizer",
            "Scheduler",
            "Best_Val_F1",
        ]
    )

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(f"Usando dispositivo: {device}")


class CroppedxBDDataset(Dataset):
    def __init__(self, data_dir, split=["train"], transform=None):
        self.data_dir = data_dir
        if isinstance(split, str):
            split = [split]
        self.split = split
        self.transform = transform
        self.image_files = []
        for s in split:
            split_dir = os.path.join(data_dir, s)
            files = natsorted([f for f in os.listdir(split_dir) if f.endswith(".png")])
            if "test" not in split:
                files = [f for f in files if not f.endswith("_-1.png")]
            self.image_files.extend([os.path.join(s, f) for f in files])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_rel_path = self.image_files[idx]
        label = int(os.path.splitext(img_rel_path)[0].split("_")[-1])
        img_path = os.path.join(self.data_dir, img_rel_path)
        image_post_np = (
            np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        )
        sample = {"patch_post": image_post_np, "label_post": label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class DictTransformWrapper:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, sample):
        image = torch.from_numpy(sample["patch_post"].transpose((2, 0, 1))).float()
        sample["patch_post"] = self.pipeline(image)
        sample["label_post"] = torch.tensor(sample["label_post"], dtype=torch.long)
        return sample


tf_basic = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

tf_advanced = transforms.Compose(
    [
        transforms.RandomCrop(64, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

tf_val = transforms.Compose(
    [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

logging.info("Cargando datasets...")
ds_train_basic = CroppedxBDDataset(
    "./xBD_cropped", ["train"], transform=DictTransformWrapper(tf_basic)
)
ds_train_adv = CroppedxBDDataset(
    "./xBD_cropped", ["train"], transform=DictTransformWrapper(tf_advanced)
)
ds_val = CroppedxBDDataset(
    "./xBD_cropped", ["val"], transform=DictTransformWrapper(tf_val)
)

loader_train_basic = DataLoader(
    ds_train_basic, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)
loader_train_adv = DataLoader(
    ds_train_adv, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)
loader_val = DataLoader(
    ds_val, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4, pin_memory=True
)

loaders = {"Basic": loader_train_basic, "Advanced": loader_train_adv}

counts = [50928, 4659, 2357, 3227]
num_classes = 4
total = sum(counts)
sqrt_weights = [math.sqrt(total / (num_classes * c)) for c in counts]
tensor_sqrt_weights = torch.tensor(sqrt_weights, dtype=torch.float32).to(device)


class CustomNet3Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CustomNet4Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_run(run_id, config, train_loader):
    logging.info(f"\n[{run_id}] Iniciando entrenamiento...")

    model = (
        CustomNet3Block().to(device)
        if config["arch"] == "3Block"
        else CustomNet4Block().to(device)
    )

    w = tensor_sqrt_weights if config["weights"] == "Sqrt" else None
    sm = 0.1 if config["loss"] == "CE_Smooth0.1" else 0.0
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=sm)

    if config["opt"] == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4
        )
    else:
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    # 4. Instanciar Scheduler
    if config["sched"] == "Step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    else:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    writer = SummaryWriter(log_dir=os.path.join(TB_DIR, f"Run_{run_id}"))
    scaler = torch.amp.GradScaler("cuda")
    best_f1 = 0.0
    best_wts = copy.deepcopy(model.state_dict())

    for epoch in range(NUM_EPOCHS):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            loader = train_loader if phase == "train" else loader_val
            running_loss = 0.0
            all_preds, all_labels = [], []

            for sample in loader:
                inputs, labels = (
                    sample["patch_post"].to(device),
                    sample["label_post"].to(device),
                )
                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == "train"):
                    with torch.amp.autocast("cuda"):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(loader.dataset)
            epoch_f1 = f1_score(all_labels, all_preds, average="macro")
            writer.add_scalar(f"Loss/{phase}", epoch_loss, epoch)
            writer.add_scalar(f"F1/{phase}", epoch_f1, epoch)

            if phase == "val" and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_wts = copy.deepcopy(model.state_dict())

        scheduler.step()

    torch.save(best_wts, os.path.join(MODELS_DIR, f"run_{run_id}.pth"))
    writer.close()

    del model, optimizer, criterion, scheduler, scaler
    torch.cuda.empty_cache()

    return best_f1


keys, values = zip(*GRID_PARAMS.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

total_runs = len(combinations)
logging.info("=== INICIANDO GRID SEARCH ===")
logging.info(f"Total de combinaciones a probar: {total_runs}")
logging.info(f"Épocas por run: {NUM_EPOCHS}")
logging.info("===================================\n")

best_global_f1 = 0.0
best_global_run = -1

for idx, config in enumerate(combinations):
    run_id = f"{idx:03d}"
    logging.info(f"Run [{run_id}/{total_runs - 1}] Config: {config}")

    start_time = time.time()

    # Ejecutar entrenamiento
    val_f1 = train_run(run_id, config, loaders[config["aug"]])

    # Tiempo
    elapsed = time.time() - start_time
    logging.info(
        f"[{run_id}] Finalizado en {elapsed / 60:.1f} min. Best Val F1: {val_f1:.4f}"
    )

    # Guardar en CSV
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                run_id,
                config["arch"],
                config["aug"],
                config["weights"],
                config["loss"],
                config["opt"],
                config["sched"],
                f"{val_f1:.4f}",
            ]
        )

    if val_f1 > best_global_f1:
        best_global_f1 = val_f1
        best_global_run = run_id
        logging.info(f"NUEVO MEJOR MODELO GLOBAL: Run {run_id} con F1: {val_f1:.4f}")
        os.system(
            f"cp {os.path.join(MODELS_DIR, f'run_{run_id}.pth')} {os.path.join(BEST_DIR, 'best_overall_model.pth')}"
        )

logging.info("\n" + "=" * 40)
logging.info("GRID SEARCH COMPLETADO.")
logging.info(
    f"El mejor modelo fue la Run {best_global_run} con un Macro F1 de {best_global_f1:.4f}"
)
logging.info("Resultados guardados en grid_training/logs/results.csv")
logging.info("========================================")
