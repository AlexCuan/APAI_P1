from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import random
import numpy.random as npr
from PIL import Image
from natsort import natsorted
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
import math
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report


random.seed(42)
npr.seed(42)
torch.manual_seed(42)

import warnings

warnings.filterwarnings("ignore")

plt.ion()  # interactive mode
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

directorio = "./xBD_UC3M"


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


# ---------------------------------------------------------
# 1. IMPROVED TRANSFORMS (Operating directly on Tensors)
# ---------------------------------------------------------
class DictTransformWrapper:
    """
    Wraps standard torchvision transforms to work with our custom dictionary dataset.
    Converts numpy arrays to tensors FIRST to avoid PIL uint8 quantization loss.
    """

    def __init__(self, transform_pipeline):
        self.pipeline = transform_pipeline

    def __call__(self, sample):
        image = sample["patch_post"]

        # Convert numpy (H x W x C) to tensor (C x H x W) in float32 [0.0, 1.0]
        image = torch.from_numpy(image.transpose((2, 0, 1))).float()

        # Apply torchvision transforms directly on the tensor
        image = self.pipeline(image)

        sample["patch_post"] = image
        sample["label_post"] = torch.tensor(sample["label_post"], dtype=torch.long)
        return sample


# Pure Tensor transformations (Much faster, preserves float32 precision)
vision_transforms_train = transforms.Compose(
    [
        transforms.RandomCrop(64, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(
            degrees=90
        ),  # 90 degrees prevents black cut-off borders on square patches
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

vision_transforms_eval = transforms.Compose(
    [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

train_transforms = DictTransformWrapper(vision_transforms_train)
eval_transforms = DictTransformWrapper(vision_transforms_eval)

# Dataset & Dataloaders
train_dataset = CroppedxBDDataset(
    "xBD_cropped", ["train"], max_size=0, transform=train_transforms
)
val_dataset = CroppedxBDDataset("xBD_cropped", ["val"], transform=eval_transforms)
test_dataset = CroppedxBDDataset("xBD_cropped", ["test"], transform=eval_transforms)

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
image_datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}


# ---------------------------------------------------------
# 2. IMPROVED ARCHITECTURE (GAP + Higher Channel Capacity)
# ---------------------------------------------------------
class CustomNetImproved(nn.Module):
    def __init__(self):
        super(CustomNetImproved, self).__init__()

        # BLOCK 1: 3 -> 32 channels
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # BLOCK 2: 32 -> 64 channels
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # BLOCK 3: 64 -> 128 channels
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global Average Pooling (Resolves the 4096-parameter bottleneck)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected (Much smaller, less prone to overfitting)
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
customNet = CustomNetImproved().to(device)

# ---------------------------------------------------------
# 3. SMOOTHED WEIGHTS & SCHEDULER SETUP
# ---------------------------------------------------------
counts = [50928, 4659, 2357, 3227]
total = sum(counts)
num_classes = 4

# Square Root Smoothing to prevent catastrophic over-penalization of minority classes
weights = [math.sqrt(total / (num_classes * c)) for c in counts]
print(f"Smoothed Class Weights[0, 1, 2, 3]: {weights}")

class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss()

# Slightly higher LR since we are using AdamW and starting from scratch

optimizer_ft = optim.SGD(
    customNet.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4
)

# ReduceLROnPlateau monitors Validation F1 and reduces LR dynamically

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)


# ---------------------------------------------------------
# 4. UPDATED TRAINING LOOP
# ---------------------------------------------------------
def train_model(
    model, criterion, optimizer, scheduler, num_epochs=25, plot_confusion_matrix=True
):
    writer = SummaryWriter()

    # Reproducibility
    import random

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    since = time.time()
    numClasses = len(image_datasets["train"].damage_classes)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    best_outputs = []
    best_labels = []

    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Print the current Learning Rate so you can see it drop!
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current Learning Rate: {current_lr:.6f}")

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
                desc=f"{phase.capitalize()} Epoch {epoch}/{num_epochs - 1}",
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

                # Store probabilities and labels
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

            # Note: We NO LONGER step the scheduler here.
            # We only evaluate if it's the best model.
            if phase == "val":
                if epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_outputs = np.argmax(outputs_m, axis=1)
                    best_labels = labels_m
                    torch.save(best_model_wts, "best_custom_net.pth")

        # STEP THE SCHEDULER HERE: Exactly once per epoch, outside the phase loop.
        scheduler.step()
        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Macro F1-score: {best_f1:.4f}")

    # Print the detailed Classification Report for the Best Epoch
    print("\nClassification Report (Best Epoch):")
    target_names = list(image_datasets["val"].damage_classes.keys())
    print(classification_report(best_labels, best_outputs, target_names=target_names))

    if plot_confusion_matrix:
        cm = confusion_matrix(best_labels, best_outputs, normalize="true")
        ncmd = ConfusionMatrixDisplay(100 * cm, display_labels=target_names)

        # Create a figure explicitly
        fig, ax = plt.subplots(figsize=(8, 6))
        ncmd.plot(xticks_rotation="vertical", cmap="Blues", ax=ax)
        plt.title("Normalized Confusion Matrix (%) - Best Epoch")

        # SAVE the file instead of (or before) showing it
        plt.savefig("best_confusion_matrix.png", bbox_inches="tight")
        print("Confusion matrix saved to 'best_confusion_matrix.png'")

        # You can comment this out if you don't want the popup at all
        # plt.show()

    model.load_state_dict(best_model_wts)
    writer.close()
    return model


# ---------------------------------------------------------
# 5. EXECUTE TRAINING
# ---------------------------------------------------------
if __name__ == "__main__":
    customNet = train_model(
        customNet,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        num_epochs=25,  # Recomended to let it run longer since you now have ReduceLROnPlateau
    )
