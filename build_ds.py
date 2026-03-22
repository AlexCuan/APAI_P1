# %%
from __future__ import print_function, division
import os
import torch
from skimage import transform, util
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import random
import numpy.random as npr
import torchvision.transforms.functional as TF
from PIL import Image
from natsort import natsorted
import json
import cv2
import tifffile
from shapely.wkt import loads
from shapely.geometry import Polygon
from tqdm import tqdm
import functools
from torch.utils.tensorboard import SummaryWriter

random.seed(42)
npr.seed(42)
torch.manual_seed(42)

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

plt.ion()  # interactive mode
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# %%
import os

# Especifica el directorio donde están las imágenes
directorio = "./xBD_UC3M"
# %% [markdown]
# ### Clase Dataset
#
# La clase ``torch.utils.data.Dataset`` es una clase abstracta que representa un dataset.
#
# Para crear nuestro propio dataset en PyTorch debemos heredar de dicha clase y sobreescribir los siguientes métodos:
#
# -  ``__init__`` el método constructor, encargado de leer e indexar la base de datos.
# -  ``__len__`` el método que permite invocar ``len(dataset)``, que nos devuelve el tamaño del dataset.
# -  ``__getitem__`` para soportar el indexado ``dataset[i]`` al referirnos a la muestra $i$.
#
# Vamos a crear los datasets de train y test de nuestro problema de evaluación de impactos de desastres naturales. Vamos a leer el csv en el método de inicialización ``__init__``, pero dejaremos la lectura explícita de las imágenes para el método
# ``__getitem__``. Esta aproximación es más eficiente en memoria porque todas las imágenes no se cargan en memoria al principio, sino que se van leyendo individualmente cuando es necesario.
#
# Cada muestra de nuestro dataset (cuando invoquemos dataset[i]) va a ser un diccionario.
#
# Por otro lado, al definir el dataset, el constructor podrá también tomar un argumento opcional ``transform`` para que podamos añadir pre-procesado y técnicas de data augmentation que le aplicaremos a las imágenes cuando las solicitemos.
# %%
from torch.utils.data import Dataset
from PIL import Image
from natsort import natsorted
import os
import numpy as np


class CroppedxBDDataset(Dataset):
    def __init__(
        self, data_dir, split=["train"], patch_size=64, transform=None, max_size=0
    ):
        self.data_dir = data_dir
        if isinstance(split, str):
            split = [split]
        self.split = split
        self.transform = transform

        self.image_files = []
        # Leemos los paths de los parches creados
        for s in split:
            split_dir = os.path.join(data_dir, s)
            # natsorted asegura que 000000, 000001 mantienen el orden estricto para el Test
            files = natsorted([f for f in os.listdir(split_dir) if f.endswith(".png")])
            if "test" not in split:
                files = [f for f in files if not f.endswith("_-1.png")]
            self.image_files.extend([os.path.join(s, f) for f in files])

        # Misma lógica de pseudo-barajado que tenías por si usas max_size
        if max_size > 0:
            idx = np.random.RandomState(seed=42).permutation(
                range(len(self.image_files))
            )
            self.image_files = [self.image_files[i] for i in idx[:max_size]]

        self.damage_classes = {
            "no-damage": 0,
            "minor-damage": 1,
            "major-damage": 2,
            "destroyed": 3,
        }

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_rel_path = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_rel_path)

        # La etiqueta la extraemos del nombre del archivo ({index}_{label}.png)
        label_str = os.path.splitext(img_rel_path)[0].split("_")[-1]
        label = int(label_str)

        # Usamos PIL para abrir rápido la imagen pre-cortada
        image_post = Image.open(img_path).convert("RGB")

        # Devolverla como array de float [0, 1] para no romper las transformaciones previas
        image_post_np = np.array(image_post, dtype=np.float32) / 255.0

        # Reconstruimos el mismo diccionario para no romper nada en el resto del notebook
        sample = {
            "patch_post": image_post_np,
            "label_post": label,
            "patch_pre": np.zeros_like(image_post_np),  # Dummy para compatibilidad
            "mask_patch": np.zeros((64, 64)),  # Dummy para compatibilidad
            "label_pre": 0,
            "idx": idx,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


# %% [markdown]
# Extraer parches de 64 x 64
# %%
# patch_size = 64
# train_dataset = CroppedxBDDataset('xBD_cropped', ['train'], patch_size=patch_size)
# %%
# val_dataset = CroppedxBDDataset('xBD_cropped', ['val'], patch_size=patch_size)
# %%
# # Elegimos un índice aleatorio
# idx = np.random.randint(0, len(train_dataset))
#
# # Obtenemos las imágenes completas originales usando la función interna
# image_pre_full, image_post_full = train_dataset._process_image(idx)
# event = train_dataset.label_post_path[idx].split('/')[-1][:-5]
#
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# fig.suptitle(f"Imágenes completas (1024x1024) - Índice: {idx} - Evento: {event}", fontsize=16)
#
# axes[0].imshow(image_pre_full)
# axes[0].set_title("Antes del desastre (Pre)")
# axes[0].axis('off')
#
# axes[1].imshow(image_post_full)
# axes[1].set_title("Después del desastre (Post)")
# axes[1].axis('off')
#
# plt.tight_layout()
# plt.show()
# %%
# # Elegimos un índice aleatorio
# idx = np.random.randint(0, len(train_dataset))
#
# # Obtenemos un ejemplo procesado directamente desde el método __getitem__
# sample = train_dataset[idx]
#
# patch_pre = sample['patch_pre']
# patch_post = sample['patch_post']
# mask_patch = sample['mask_patch']
# label_post = sample['label_post']
#
# # Mapeo inverso de las clases de daño para la visualización
# inverse_damage_classes = {v: k for k, v in train_dataset.damage_classes.items()}
# damage_text = inverse_damage_classes[label_post]
#
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# fig.suptitle(f"Parche de Edificio (64x64) - Etiqueta Post-Desastre: {damage_text} ({label_post})", fontsize=16)
#
# # Aseguramos que los valores estén en el rango correcto para matplotlib (0-255 o 0-1)
# axes[0].imshow(patch_pre)
# axes[0].set_title("Parche Pre-desastre")
# axes[0].axis('off')
#
# axes[1].imshow(patch_post)
# axes[1].set_title("Parche Post-desastre")
# axes[1].axis('off')
#
# axes[2].imshow(mask_patch, cmap='gray')
# axes[2].set_title("Máscara del Edificio")
# axes[2].axis('off')
#
# plt.tight_layout()
# plt.show()
# %%
class RandomCrop(object):
    """Recortamos aleatoriamente la imagen.

    Args:
        output_size (tupla o int): Tamaño del recorte. Si int, recorte cuadrado

    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample["patch_post"]

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        if h > new_h:
            top = np.random.randint(0, h - new_h)
        else:
            top = 0

        if w > new_w:
            left = np.random.randint(0, w - new_w)
        else:
            left = 0

        image = image[top : top + new_h, left : left + new_w]

        sample = {
            "patch_pre": sample["patch_pre"],
            "patch_post": image,
            "mask_patch": sample["mask_patch"],
            "label_pre": sample["label_pre"],
            "label_post": sample["label_post"],
            "idx": sample["idx"],
        }
        return sample


class CenterCrop(object):
    """Crop the central area of the image

    Args:
        output_size (tupla or int): Crop size. If int, square crop

    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample["patch_post"]
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        rem_h = h - new_h
        rem_w = w - new_w

        if h > new_h:
            top = int(rem_h / 2)
        else:
            top = 0

        if w > new_w:
            left = int(rem_w / 2)
        else:
            left = 0

        image = image[top : top + new_h, left : left + new_w]

        sample = {
            "patch_pre": sample["patch_pre"],
            "patch_post": image,
            "mask_patch": sample["mask_patch"],
            "label_pre": sample["label_pre"],
            "label_post": sample["label_post"],
            "idx": sample["idx"],
        }
        return sample


class Rescale(object):
    """Re-escalamos la imagen a un tamaño determinado.

    Args:
        output_size (tupla o int): El tamaño deseado. Si es una tupla, output es el output_size.
        Si es un int, la dimensión más pequeña será el output_size
            y mantendremos la relación de aspecto original.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample["patch_post"]

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        sample = {
            "patch_pre": sample["patch_pre"],
            "patch_post": image,
            "mask_patch": sample["mask_patch"],
            "label_pre": sample["label_pre"],
            "label_post": sample["label_post"],
            "idx": sample["idx"],
        }
        return sample


class ToTensor(object):
    """Convertimos ndarrays de la muestra en tensores."""

    def __call__(self, sample):
        image, label = sample["patch_post"], sample["label_post"]

        # Cambiamos los ejes
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image.astype(np.float32))

        label = torch.tensor(label, dtype=torch.long)

        sample = {
            "patch_pre": sample["patch_pre"],
            "patch_post": image,
            "mask_patch": sample["mask_patch"],
            "label_pre": sample["label_pre"],
            "label_post": label,
            "idx": sample["idx"],
        }
        return sample


class Normalize(object):
    """Normalizamos los datos restando la media y dividiendo por las desviaciones típicas.

    Args:
        mean_vec: El vector con las medias.
        std_vec: el vector con las desviaciones típicas.
    """

    def __init__(self, mean, std):

        assert len(mean) == len(std), "Length of mean and std vectors is not the same"
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, sample):
        image = sample["patch_post"]
        c, h, w = image.shape
        assert c == len(self.mean), "Length of mean and image is not the same"
        dtype = image.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=image.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=image.device)
        image.sub_(mean[:, None, None]).div_(std[:, None, None])

        sample = {
            "patch_pre": sample["patch_pre"],
            "patch_post": image,
            "mask_patch": sample["mask_patch"],
            "label_pre": sample["label_pre"],
            "label_post": sample["label_post"],
            "idx": sample["idx"],
        }
        return sample


# %%
class AugmentPostPatch(object):
    """
    Extrae 'patch_post' del diccionario, le aplica transformaciones de torchvision
    y lo devuelve al diccionario.
    """

    def __init__(self, transform_pipeline):
        self.transform_pipeline = transform_pipeline

    def __call__(self, sample):
        image = sample["patch_post"]

        # 1. Convertimos la matriz numpy a imagen PIL (formato de 8 bits: 0-255)
        pil_image = Image.fromarray(util.img_as_ubyte(image))

        # 2. Aplicamos el pipeline de transformaciones (ej. flips, rotaciones)
        transformed_image = self.transform_pipeline(pil_image)

        # 3. Volvemos a convertir a matriz numpy flotante (0.0 a 1.0)
        image_np = util.img_as_float(np.asarray(transformed_image))

        # 4. Actualizamos el diccionario y lo devolvemos
        sample["patch_post"] = image_np
        return sample


# %%
vision_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(
            p=0.5
        ),  # Voltea la imagen horizontalmente (50% de prob)
        transforms.RandomVerticalFlip(
            p=0.5
        ),  # Voltea la imagen verticalmente (50% de prob)
        transforms.RandomRotation(
            degrees=45
        ),  # Rota el edificio aleatoriamente hasta 45 grados
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        ),  # Simula cambios de iluminación/clima
    ]
)

# Componemos el pipeline completo para Train (Aumentación + Tensores + Normalización)
train_transforms = transforms.Compose(
    [
        AugmentPostPatch(vision_transforms),  # 1. Aplicamos data augmentation
        ToTensor(),  # 2. Convertimos a tensor
        Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # 3. Normalizamos
    ]
)

# Pipeline para Validación y Test sin augmentation
eval_transforms = transforms.Compose(
    [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


# Datos de train
train_dataset = CroppedxBDDataset(
    "xBD_cropped", ["train"], max_size=0, transform=train_transforms
)

# Datos de validación
val_dataset = CroppedxBDDataset("xBD_cropped", ["val"], transform=eval_transforms)

# Datos de test
test_dataset = CroppedxBDDataset("xBD_cropped", ["test"], transform=eval_transforms)
# %%
train_dataloader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
# %%
import torch.nn as nn
import torch.nn.functional as F


class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()

        # -------------------------------------------------------------------
        # BLOCK 1
        # Input shape:[Batch_size, 3, 64, 64]
        # -------------------------------------------------------------------
        # In VGG, we use kernel_size=3 and padding=1 so the spatial dimensions
        # don't change after a convolution, only after MaxPool.
        self.conv1_1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1
        )
        self.bn1_1 = nn.BatchNorm2d(16)  # Batch Normalization accelerates training
        self.conv1_2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, padding=1
        )
        self.bn1_2 = nn.BatchNorm2d(16)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output shape after pool: [Batch_size, 16, 32, 32]

        # -------------------------------------------------------------------
        # BLOCK 2
        # Input shape:[Batch_size, 16, 32, 32]
        # -------------------------------------------------------------------
        self.conv2_1 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.bn2_1 = nn.BatchNorm2d(32)
        self.conv2_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1
        )
        self.bn2_2 = nn.BatchNorm2d(32)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output shape after pool:[Batch_size, 32, 16, 16]

        # -------------------------------------------------------------------
        # BLOCK 3
        # Input shape: [Batch_size, 32, 16, 16]
        # -------------------------------------------------------------------
        self.conv3_1 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn3_1 = nn.BatchNorm2d(64)
        self.conv3_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )
        self.bn3_2 = nn.BatchNorm2d(64)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output shape after pool: [Batch_size, 64, 8, 8]

        # -------------------------------------------------------------------
        # FULLY CONNECTED CLASSIFIER
        # -------------------------------------------------------------------
        # Flattening 64 channels of 8x8 spatial dimensions = 64 * 8 * 8 = 4096
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=256)

        # Dropout is highly recommended for VGG-style networks to prevent
        # overfitting in the dense layers, especially with small datasets.
        self.dropout = nn.Dropout(p=0.5)

        # Output layer: 4 classes (No Damage, Minor, Major, Destroyed)
        self.fc2 = nn.Linear(in_features=256, out_features=4)

    def forward(self, x):
        # Forward pass for Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)

        # Forward pass for Block 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)

        # Forward pass for Block 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)

        # Flatten for the fully connected layers
        x = x.flatten(1)

        # Fully connected phase
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout during training
        x = self.fc2(x)  # Produce the final 4 logits

        return x


# %%
customNet = CustomNet()  # Invocamos el constructor de la red (método init())
customNet.to(device)  # Pasamos la red al device que estemos usando (gpu)
# Obtenemos un batch de datos y extraemos imágenes y etiquetas
customNet = torch.compile(customNet)

data = next(iter(train_dataloader))
inputs = data["patch_post"].to(device).float()
labels = data["label_post"].to(device)

batchSize = labels.shape
print(
    "El tamaño del tensor que representa un batch de imágenes es {}".format(
        inputs.shape
    )
)

# Lo pasamos por la red
with torch.set_grad_enabled(False):
    outputs = customNet(inputs)
    print("El tamaño del tensor de salida es {}".format(outputs.shape))
# %%
counts = [50928, 4659, 2357, 3227]
total = sum(counts)
num_classes = 4

# 2. Calculamos los pesos balanceados (fórmula clásica: total / (num_clases * count))
# Esto dará menos peso a la clase mayoritaria (0.3) y mucho a las minoritarias (hasta 6.4)
weights = [total / (num_classes * c) for c in counts]
print(f"Pesos asignados a las clases[0, 1, 2, 3]: {weights}")

# Convertimos a tensor de PyTorch y lo mandamos a la GPU
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

# 3. Le pasamos los pesos a la función de pérdida
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Usaremos SGD con momento para optimizar
optimizer_ft = optim.AdamW(customNet.parameters(), lr=1e-4, weight_decay=1e-2)

# Un factor lr que  decae 0.1 cada 7 épocas
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.5)
# %%
from tqdm import tqdm  # Asegúrate de tenerlo importado


def train_model(
    model, criterion, optimizer, scheduler, num_epochs=25, plot_confusion_matrix=True
):

    writer = SummaryWriter()

    # Fijamos semillas para maximizar reproducibilidad
    random.seed(42)
    npr.seed(42)
    torch.manual_seed(42)

    # 1. OPTIMIZACIÓN: cuDNN Benchmark en True (Acelera CNNs en GPUs NVIDIA)
    torch.backends.cudnn.benchmark = True

    since = time.time()

    numClasses = len(list(image_datasets["train"].damage_classes.keys()))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0
    best_outputs = []
    best_labels = []

    # 2. OPTIMIZACIÓN: Inicializamos el GradScaler para Mixed Precision (AMP)
    scaler = torch.amp.GradScaler("cuda")

    # Bucle de épocas de entrenamiento
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Cada época tiene entrenamiento y validación
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Ponemos el modelo en modo entrenamiento
            else:
                model.eval()  # Ponemos el modelo en modo evaluación

            # Tamaño del dataset
            numSamples = dataset_sizes[phase]

            # Creamos las variables que almacenarán las salidas y las etiquetas
            outputs_m = np.zeros((numSamples, numClasses), dtype=np.float32)
            labels_m = np.zeros((numSamples,), dtype=int)
            running_loss = 0.0

            contSamples = 0

            # ------------------------------------------------------------------
            # NUEVO: Integramos tqdm para la barra de progreso
            # leave=False hace que la barra desaparezca al llegar al 100%
            # para no saturar la pantalla (dejando solo los prints limpios)
            # ------------------------------------------------------------------
            progress_bar = tqdm(
                dataloaders[phase],
                desc=f"{phase.capitalize()} Epoch {epoch}/{num_epochs - 1}",
                leave=False,
            )

            # Iteramos sobre los datos usando progress_bar en lugar de dataloaders[phase]
            for sample in progress_bar:
                inputs = sample["patch_post"].to(device)
                labels = sample["label_post"].to(device)

                # Tamaño del batch
                batchSize = labels.shape[0]

                # 3. OPTIMIZACIÓN: set_to_none=True es más rápido y consume menos VRAM
                optimizer.zero_grad(set_to_none=True)

                # Paso forward
                # registramos operaciones solo en train
                with torch.set_grad_enabled(phase == "train"):
                    # 4. OPTIMIZACIÓN: Autocast para Mixed Precision
                    with torch.amp.autocast("cuda"):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    # backward y optimización solo en training usando el Scaler
                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                # Sacamos estadísticas y actualizamos variables
                running_loss += loss.item() * inputs.size(0)

                # Aplicamos un softmax a la salida
                # IMPORTANTE: Pasamos a float() porque con AMP la salida estará en float16
                outputs_probs = F.softmax(outputs.detach(), dim=1).float()

                outputs_m[contSamples : contSamples + batchSize, ...] = (
                    outputs_probs.cpu().numpy()
                )
                labels_m[contSamples : contSamples + batchSize] = labels.cpu().numpy()
                contSamples += batchSize

                # NUEVO: Actualizamos el texto a la derecha de la barra con la Loss actual
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            # Actualizamos la estrategia de lr
            if phase == "train":
                scheduler.step()

            # Loss acumulada en la época
            epoch_loss = running_loss / dataset_sizes[phase]

            # Calculamos Macro F1-score
            epoch_f1 = f1_score(labels_m, np.argmax(outputs_m, axis=1), average="macro")
            writer.add_scalar(f"Loss/{phase}", epoch_loss, epoch)
            writer.add_scalar(f"F1_Score/{phase}", epoch_f1, epoch)

            print(
                "{} Loss: {:.4f} Macro F1-score: {:.4f}".format(
                    phase, epoch_loss, epoch_f1
                )
            )

            # copia profunda del mejor modelo
            if phase == "val" and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())
                best_outputs = np.argmax(outputs_m, axis=1)
                best_labels = labels_m

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Macro F1-score: {:4f}".format(best_f1))

    if plot_confusion_matrix:
        # Visualizamos la matriz de confusión
        cm = confusion_matrix(best_labels, best_outputs, normalize="true")
        # Vamos a mostrar en porcentajes en vez de probs
        ncmd = ConfusionMatrixDisplay(
            100 * cm, display_labels=list(image_datasets["val"].damage_classes.keys())
        )
        ncmd.plot(xticks_rotation="vertical")
        plt.title("Normalized confusion matrix (%)")
        plt.show()

    # load best model weights
    model.load_state_dict(best_model_wts)
    writer.close()
    return model


# %%
image_datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

dataloaders = {
    "train": train_dataloader,
    "val": val_dataloader,
    "test": test_dataloader,
}

dataset_sizes = {
    "train": len(train_dataset),
    "val": len(val_dataset),
    "test": len(test_dataset),
}
class_names = list(image_datasets["train"].damage_classes.keys())
# %%
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# %%
customNet = train_model(
    customNet, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20
)
# %%
