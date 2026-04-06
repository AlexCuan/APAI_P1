import os
import time
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torch.utils.tensorboard.writer import SummaryWriter
from PIL import Image
from natsort import natsorted
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

import warnings

warnings.filterwarnings("ignore")

plt.ion()
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

        image_post = Image.open(img_path).convert("RGB")
        image_post_np = np.array(image_post, dtype=np.float32) / 255.0

        sample = {"patch_post": image_post_np, "label_post": label, "idx": idx}
        if self.transform:
            sample = self.transform(sample)
        return sample


class DictTransformWrapper:
    def __init__(self, transform_pipeline):
        self.pipeline = transform_pipeline

    def __call__(self, sample):
        image = sample["patch_post"]
        image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        sample["patch_post"] = self.pipeline(image)
        sample["label_post"] = torch.tensor(sample["label_post"], dtype=torch.long)
        return sample


vision_transforms_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

vision_transforms_eval = transforms.Compose(
    [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

train_transforms = DictTransformWrapper(vision_transforms_train)
eval_transforms = DictTransformWrapper(vision_transforms_eval)

print("\nCargando datasets...")
train_dataset = CroppedxBDDataset("xBD_cropped", ["train"], transform=train_transforms)
val_dataset = CroppedxBDDataset("xBD_cropped", ["val"], transform=eval_transforms)

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

dataloaders = {"train": train_dataloader, "val": val_dataloader}
dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}


class CustomNetImproved(nn.Module):
    def __init__(self):
        super(CustomNetImproved, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool1(
            F.relu(self.bn1_2(self.conv1_2(F.relu(self.bn1_1(self.conv1_1(x))))))
        )
        x = self.pool2(
            F.relu(self.bn2_2(self.conv2_2(F.relu(self.bn2_1(self.conv2_1(x))))))
        )
        x = self.pool3(
            F.relu(self.bn3_2(self.conv3_2(F.relu(self.bn3_1(self.conv3_1(x))))))
        )
        x = self.gap(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


customNet = CustomNetImproved().to(device)

num_epochs = 25

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.AdamW(customNet.parameters(), lr=1e-3, weight_decay=1e-2)

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    writer = SummaryWriter(log_dir="runs/custom_oversampled")
    since = time.time()
    numClasses = 4

    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    best_outputs, best_labels = [], []
    scaler = torch.amp.GradScaler("cuda")

    print("\nIniciando entrenamiento...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs - 1}")
        print("-" * 10)
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            numSamples = dataset_sizes[phase]
            outputs_m = np.zeros((numSamples, numClasses), dtype=np.float32)
            labels_m = np.zeros((numSamples,), dtype=int)
            running_loss = 0.0
            contSamples = 0

            progress_bar = tqdm(
                dataloaders[phase],
                desc=f"{phase.capitalize()} Epoch {epoch}",
                leave=False,
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
                f"{phase.capitalize()} Loss: {epoch_loss:.4f} Macro F1-score: {epoch_f1:.4f}"
            )

            if phase == "val":
                if epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_outputs = np.argmax(outputs_m, axis=1)
                    best_labels = labels_m
                    torch.save(best_model_wts, "best_custom_net.pth")
                    print(">>> ¡Nuevo mejor modelo guardado (best_custom_net.pth)!")

        scheduler.step()

    time_elapsed = time.time() - since
    print(
        f"\nEntrenamiento completo en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
    )
    print(f"Mejor Macro F1-score de Validación: {best_f1:.4f}")

    target_names = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    print("\nClassification Report (Best Epoch):")
    print(classification_report(best_labels, best_outputs, target_names=target_names))

    cm = confusion_matrix(best_labels, best_outputs, normalize="true")
    ncmd = ConfusionMatrixDisplay(100 * cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    ncmd.plot(xticks_rotation="vertical", cmap="Blues", ax=ax)
    plt.title("Normalized Confusion Matrix (%) - CustomNet Oversampled")
    plt.savefig("best_confusion_matrix.png", bbox_inches="tight")
    print("Matriz de confusión guardada como 'best_confusion_matrix.png'")

    model.load_state_dict(best_model_wts)
    writer.close()
    return model


if __name__ == "__main__":
    customNet = train_model(customNet, criterion, optimizer, scheduler, num_epochs=25)
