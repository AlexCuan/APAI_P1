import os
import time
import copy
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torch.utils.tensorboard.writer import SummaryWriter
from PIL import Image
from natsort import natsorted
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.benchmark = True


class CroppedxBDDataset(Dataset):
    def __init__(self, data_dir, split=["train"], transform=None, max_size=0):
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
        label = int(os.path.splitext(img_rel_path)[0].split("_")[-1])
        image_post_np = (
            np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        )
        sample = {"patch_post": image_post_np, "label_post": label, "idx": idx}
        if self.transform:
            sample = self.transform(sample)
        return sample


class DictTransformWrapper:
    def __init__(self, transform_pipeline):
        self.pipeline = transform_pipeline

    def __call__(self, sample):
        image = torch.from_numpy(sample["patch_post"].transpose((2, 0, 1))).float()
        sample["patch_post"] = self.pipeline(image)
        sample["label_post"] = torch.tensor(sample["label_post"], dtype=torch.long)
        return sample


tf_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

tf_val = transforms.Compose(
    [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

train_dataset = CroppedxBDDataset(
    "xBD_cropped", ["train"], transform=DictTransformWrapper(tf_train)
)
val_dataset = CroppedxBDDataset(
    "xBD_cropped", ["val"], transform=DictTransformWrapper(tf_val)
)

print("\nCalculando pesos para el Oversampling dinámico...")
train_labels = [
    int(os.path.splitext(f)[0].split("_")[-1]) for f in train_dataset.image_files
]
class_counts = np.bincount(train_labels)
print(f"Distribución original en Train: {class_counts}")

class_weights = 1.0 / class_counts
sample_weights = [class_weights[label] for label in train_labels]
sample_weights = torch.DoubleTensor(sample_weights)

sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(sample_weights), replacement=True
)

train_dataloader = DataLoader(
    train_dataset, batch_size=128, sampler=sampler, num_workers=4, pin_memory=True
)
val_dataloader = DataLoader(
    val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
)

dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}

print("\nInicializando VGG11_bn pre-entrenada...")
ftNet = models.vgg11_bn(weights=models.VGG11_BN_Weights.DEFAULT)

ftNet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
ftNet.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(512, 128),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(128, 4),
)

ftNet = ftNet.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


def train_finetune(model, criterion, num_epochs=15):
    writer = SummaryWriter(log_dir="runs/finetune_vgg11_oversampled")
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    best_outputs, best_labels = [], []
    scaler = torch.amp.GradScaler("cuda")

    print("\n>>> FASE 1: Congelando Backbone... Entrenando solo el clasificador.")
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        if epoch == 2:
            print("\n>>> FASE 2: Descongelando Backbone para Fine-Tuning completo...")
            for param in model.parameters():
                param.requires_grad = True

            backbone_params = [
                p for n, p in model.named_parameters() if "classifier" not in n
            ]
            head_params = model.classifier.parameters()

            optimizer = optim.AdamW(
                [
                    {"params": backbone_params, "lr": 1e-5},
                    {"params": head_params, "lr": 1e-4},
                ],
                weight_decay=1e-2,
            )

            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=(num_epochs - 2)
            )

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            outputs_m = np.zeros((dataset_sizes[phase], 4), dtype=np.float32)
            labels_m = np.zeros((dataset_sizes[phase],), dtype=int)
            running_loss = 0.0
            contSamples = 0

            loader = train_dataloader if phase == "train" else val_dataloader
            progress_bar = tqdm(
                loader, desc=f"{phase.capitalize()} Epoch {epoch}", leave=False
            )

            for sample in progress_bar:
                inputs = sample["patch_post"].to(device)
                labels = sample["label_post"].to(device)
                batchSize = labels.size(0)

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
                outputs_probs = F.softmax(outputs.detach(), dim=1).float()
                outputs_m[contSamples : contSamples + batchSize, ...] = (
                    outputs_probs.cpu().numpy()
                )
                labels_m[contSamples : contSamples + batchSize] = labels.cpu().numpy()
                contSamples += batchSize
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_f1 = f1_score(labels_m, np.argmax(outputs_m, axis=1), average="macro")

            writer.add_scalar(f"Loss/{phase}", epoch_loss, epoch)
            writer.add_scalar(f"F1_Score/{phase}", epoch_f1, epoch)

            print(
                f"{phase.capitalize()} Loss: {epoch_loss:.4f} Macro F1: {epoch_f1:.4f}"
            )

            if phase == "val":
                if epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_outputs = np.argmax(outputs_m, axis=1)
                    best_labels = labels_m
                    torch.save(best_model_wts, "best_ft_net.pth")
                    print(
                        ">>> ¡Nuevo mejor modelo Fine-Tuned guardado (best_ft_net.pth)!"
                    )

        scheduler.step()

    time_elapsed = time.time() - since
    print(
        f"\nFine-Tuning completo en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
    )
    print(f"Mejor Macro F1-score de Validación: {best_f1:.4f}")

    target_names = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    print("\nClassification Report (Best Epoch):")
    print(classification_report(best_labels, best_outputs, target_names=target_names))

    cm = confusion_matrix(best_labels, best_outputs, normalize="true")
    ncmd = ConfusionMatrixDisplay(100 * cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    ncmd.plot(xticks_rotation="vertical", cmap="Blues", ax=ax)
    plt.title("Normalized Confusion Matrix (%) - Fine-Tuning VGG11_bn")
    plt.savefig("best_ft_confusion_matrix.png", bbox_inches="tight")
    print("Matriz de confusión guardada como 'best_ft_confusion_matrix.png'")

    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    ftNet = train_finetune(ftNet, criterion, num_epochs=15)
