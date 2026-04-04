import os
import csv
from zipfile import ZipFile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
import json
import cv2
import tifffile
from shapely.wkt import loads
from shapely.geometry import Polygon
from natsort import natsorted
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ToTensor(object):
    def __call__(self, sample):
        image, label = sample["patch_post"], sample["label_post"]
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.tensor(label, dtype=torch.long)
        sample = {"patch_post": image, "label_post": label, "idx": sample["idx"]}
        return sample


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, sample):
        image = sample["patch_post"]
        dtype = image.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=image.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=image.device)
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        sample["patch_post"] = image
        return sample


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class xBDDataset(Dataset):
    def __init__(
        self, data_dir, split=["test"], patch_size=64, transform=None, max_size=0
    ):
        self.data_dir = data_dir
        self.split = split
        self.patch_size = patch_size
        self.transform = transform
        self.max_size = max_size

        self.image_pre_files, self.image_post_files = [], []
        self.label_pre_files, self.label_post_files = [], []

        for s in split:
            disaster_dirs = os.listdir(os.path.join(data_dir, s))
            for d in disaster_dirs:
                image_files = os.listdir(os.path.join(data_dir, s, d, "images"))
                labels_files = os.listdir(os.path.join(data_dir, s, d, "labels"))
                self.image_pre_files.append(
                    natsorted(
                        [
                            os.path.join(data_dir, s, d, "images", f)
                            for f in image_files
                            if "pre" in f
                        ]
                    )
                )
                self.image_post_files.append(
                    natsorted(
                        [
                            os.path.join(data_dir, s, d, "images", f)
                            for f in image_files
                            if "post" in f
                        ]
                    )
                )
                self.label_pre_files.append(
                    natsorted(
                        [
                            os.path.join(data_dir, s, d, "labels", f)
                            for f in labels_files
                            if "pre" in f
                        ]
                    )
                )
                self.label_post_files.append(
                    natsorted(
                        [
                            os.path.join(data_dir, s, d, "labels", f)
                            for f in labels_files
                            if "post" in f
                        ]
                    )
                )

        self.image_pre_files = natsorted(np.hstack(self.image_pre_files))
        self.image_post_files = natsorted(np.hstack(self.image_post_files))
        self.label_pre_files = natsorted(np.hstack(self.label_pre_files))
        self.label_post_files = natsorted(np.hstack(self.label_post_files))

        self.damage_classes = {
            "no-damage": 0,
            "minor-damage": 1,
            "major-damage": 2,
            "destroyed": 3,
        }
        self._get_patches_data()

    def _process_image(self, idx):
        image_post = tifffile.imread(self.patch_image_post[idx])
        if image_post.shape[:2] != (1024, 1024):
            image_post = cv2.resize(image_post, (1024, 1024))
        return None, image_post

    def _get_patches_data(self):
        num_files = len(self.image_pre_files)
        self.patch_post = []
        self.label_post = []
        self.patch_image_post = []

        for n in np.arange(num_files):
            label_post_path = self.label_post_files[n]
            with open(label_post_path, "r") as f_post:
                data_post = json.load(f_post)
                if "features" in data_post and "xy" in data_post["features"]:
                    features_post = data_post["features"]["xy"]
                    for feature in features_post:
                        if "wkt" in feature and "properties" in feature:
                            props = feature["properties"]
                            if props.get("feature_type") == "building":
                                damage_type = props.get("subtype", "no-damage")
                                if (
                                    damage_type in self.damage_classes
                                    or damage_type == -1
                                ):
                                    self.patch_post.append(feature["wkt"])
                                    self.label_post.append(
                                        self.damage_classes[damage_type]
                                        if damage_type in self.damage_classes
                                        else -1
                                    )
                                    self.patch_image_post.append(
                                        self.image_post_files[n]
                                    )

    def _extract_patch(self, image, wkt_str):
        geom = loads(wkt_str)
        if isinstance(geom, Polygon) and geom.is_valid:
            coords = np.array(geom.exterior.coords, dtype=np.int32)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            half_size = self.patch_size // 2

            x1 = max(0, center_x - half_size)
            y1 = max(0, center_y - half_size)
            x2 = min(image.shape[1], center_x + half_size)
            y2 = min(image.shape[0], center_y + half_size)

            patch = image[y1:y2, x1:x2]
            if patch.shape[0] > 0 and patch.shape[1] > 0:
                patch = cv2.resize(patch, (self.patch_size, self.patch_size))
        else:
            patch = np.zeros((self.patch_size, self.patch_size, image.shape[-1]))
        return patch

    def __len__(self):
        return len(self.patch_post)

    def __getitem__(self, idx):
        _, image_post = self._process_image(idx)
        image_post = image_post / 255.0
        patch_post = self._extract_patch(image_post, self.patch_post[idx])

        sample = {
            "patch_post": patch_post,
            "label_post": self.label_post[idx],
            "idx": idx,
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


class CustomNetImproved(nn.Module):
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


def get_finetuned_net():
    net = models.vgg11_bn(weights=None)
    net.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    net.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(128, 4),
    )
    return net


def test_model(model, dataloader, num_samples):
    model.eval()
    outputs_m = np.zeros((num_samples, 4), dtype=np.float32)
    contSamples = 0

    progress_bar = tqdm(dataloader, desc="Running Inference", leave=False)
    for sample in progress_bar:
        inputs = sample["patch_post"].to(device).float()
        batchSize = inputs.shape[0]

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
            outputs = F.softmax(outputs.data, dim=1)
            outputs_m[contSamples : contSamples + batchSize, ...] = (
                outputs.cpu().numpy()
            )
            contSamples += batchSize

    return outputs_m


if __name__ == "__main__":
    print("Cargando Test Dataset Original")

    ORIGINAL_DATA_DIR = "../xBD_UC3M"

    test_transform = Compose(
        [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    test_dataset = xBDDataset(
        ORIGINAL_DATA_DIR, ["test"], patch_size=64, transform=test_transform
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=4
    )
    NUM_TEST_SAMPLES = len(test_dataset)

    print(
        f"\nImágenes de Test cargadas correctamente: {NUM_TEST_SAMPLES} (Debe ser exactamente 22222)"
    )

    outputs_custom = np.zeros((NUM_TEST_SAMPLES, 4), dtype=np.float32)
    outputs_ft = np.zeros((NUM_TEST_SAMPLES, 4), dtype=np.float32)

    print("\nTesting Custom Net...")
    if os.path.exists("best_custom_net.pth"):
        customNet = CustomNetImproved().to(device)
        customNet.load_state_dict(
            torch.load("best_custom_net.pth", map_location=device)
        )
        outputs_custom = test_model(customNet, test_dataloader, NUM_TEST_SAMPLES)
    else:
        print("WARNING: 'best_custom_net.pth' no encontrado. Rellenando con ceros.")

    print("\nTesting Fine-Tuned Net...")
    if os.path.exists("best_ft_net.pth"):
        ftNet = get_finetuned_net().to(device)
        ftNet.load_state_dict(torch.load("best_ft_net.pth", map_location=device))
        outputs_ft = test_model(ftNet, test_dataloader, NUM_TEST_SAMPLES)
    else:
        print("WARNING: 'best_ft_net.pth' no encontrado. Rellenando con ceros.")

    print("\nGenerando archivos CSV y archivo ZIP...")

    with open("output_custom.csv", mode="w", newline="") as out_file:
        csv_writer = csv.writer(out_file, delimiter=",")
        csv_writer.writerows(outputs_custom)

    with open("output_ft.csv", mode="w", newline="") as out_file:
        csv_writer = csv.writer(out_file, delimiter=",")
        csv_writer.writerows(outputs_ft)

    with ZipFile("./codabench_submission.zip", "w") as zip_object:
        zip_object.write("./output_custom.csv")
        zip_object.write("./output_ft.csv")

    print("\ncodabench_submission.zip generado con éxito.")
