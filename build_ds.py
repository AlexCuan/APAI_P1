#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

random.seed(42)
npr.seed(42)
torch.manual_seed(42)

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[2]:


import os

# Especifica el directorio donde están las imágenes
directorio = "./xBD_UC3M"


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

# In[3]:


class xBDDataset(Dataset):
    def __init__(self, data_dir, split=['train'], patch_size=64, transform=None, max_size=0):
        self.data_dir = data_dir
        self.split = split
        self.patch_size = patch_size
        self.transform = transform
        self.max_size = max_size

        self.image_pre_files = []
        self.image_post_files = []
        self.label_pre_files = []
        self.label_post_files = []
        for s in split:
            disaster_dirs = os.listdir(os.path.join(data_dir, s))

            for d in disaster_dirs:
                image_files = os.listdir(os.path.join(data_dir, s, d, 'images'))
                labels_files = os.listdir(os.path.join(data_dir, s, d, 'labels'))
                self.image_pre_files.append(natsorted([os.path.join(data_dir, s, d, 'images', f) \
                                        for f in image_files if 'pre' in f]))
                self.image_post_files.append(natsorted([os.path.join(data_dir, s, d, 'images', f) \
                                        for f in image_files if 'post' in f]))
                self.label_pre_files.append(natsorted([os.path.join(data_dir, s, d, 'labels', f) \
                                        for f in labels_files if 'pre' in f]))
                self.label_post_files.append(natsorted([os.path.join(data_dir, s, d, 'labels', f) \
                                        for f in labels_files if 'post' in f]))
        self.image_pre_files = natsorted(np.hstack(self.image_pre_files))
        self.image_post_files = natsorted(np.hstack(self.image_post_files))
        self.label_pre_files = natsorted(np.hstack(self.label_pre_files))
        self.label_post_files = natsorted(np.hstack(self.label_post_files))

        print(split)
        self.damage_classes = {
            'no-damage': 0,
            'minor-damage': 1,
            'major-damage': 2,
            'destroyed': 3
        }

        self.patches = []
        self.patch_labels = []
        self._get_patches_data()
        self._print_class_distribution()

    def _process_image(self, idx):
        image_pre = tifffile.imread(self.patch_image_pre[idx])
        if image_pre.shape[:2] != (1024, 1024):
            image_pre = cv2.resize(image_pre, (1024, 1024))

        image_post = tifffile.imread(self.patch_image_post[idx])
        if image_post.shape[:2] != (1024, 1024):
            image_post = cv2.resize(image_post, (1024, 1024))

        return image_pre, image_post #, mask_pre

    def _create_mask(self, label_path):
        mask = np.zeros((1024, 1024), dtype=np.uint8)

        try:
            with open(label_path, 'r') as f:
                data = json.load(f)

            if 'features' in data and 'xy' in data['features']:
                features = data['features']['xy']

                for feature in features:
                    if 'wkt' in feature:
                        try:
                            geom = loads(feature['wkt'])
                            if isinstance(geom, Polygon) and geom.is_valid:
                                coords = np.array(geom.exterior.coords, dtype=np.int32)
                                cv2.fillPoly(mask, [coords], 1)
                        except:
                            continue
        except:
            pass

        return mask

    def _get_patches_data(self):
        num_files = len(self.image_pre_files)
        self.patch_pre = []
        self.patch_post = []
        self.label_pre = []
        self.label_post = []
        self.label_post_path = []
        self.patch_image_pre = []
        self.patch_image_post = []

        for n in np.arange(num_files):
            label_pre_path = self.label_pre_files[n]
            label_post_path = self.label_post_files[n]

            with open(label_pre_path, 'r') as f_pre, open(label_post_path, 'r') as f_post:
                data_pre = json.load(f_pre)
                data_post = json.load(f_post)

                if 'features' in data_post and 'xy' in data_post['features']:
                    features_post = data_post['features']['xy']

                    for feature in features_post:
                        if 'wkt' in feature and 'properties' in feature:
                            props = feature['properties']
                            if props.get('feature_type') == 'building':
                                damage_type = props.get('subtype', 'no-damage')

                                if damage_type in self.damage_classes or damage_type == -1:
                                    self.patch_pre.append(feature['wkt'])
                                    self.patch_post.append(feature['wkt'])
                                    self.label_pre.append(0) # No damage before the event
                                    self.label_post.append(self.damage_classes[damage_type] if damage_type in self.damage_classes else -1)
                                    self.label_post_path.append(label_post_path)
                                    self.patch_image_pre.append(self.image_pre_files[n])
                                    self.patch_image_post.append(self.image_post_files[n])

        if self.max_size > 0:
            new_dataset_size = self.max_size
            idx = np.random.RandomState(seed=42).permutation(range(len(self.patch_pre)))
            self.patch_pre = np.array(self.patch_pre)[idx[0:new_dataset_size]]
            self.patch_post = np.array(self.patch_post)[idx[0:new_dataset_size]]
            self.label_pre = np.array(self.label_pre)[idx[0:new_dataset_size]]
            self.label_post = np.array(self.label_post)[idx[0:new_dataset_size]]
            self.label_post_path = np.array(self.label_post_path)[idx[0:new_dataset_size]]
            self.patch_image_pre = np.array(self.patch_image_pre)[idx[0:new_dataset_size]]
            self.patch_image_post = np.array(self.patch_image_post)[idx[0:new_dataset_size]]

    def _extract_patch(self, image, mask, wkt_str):
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
            if mask is not None:
                mask_patch = mask[y1:y2, x1:x2]
            else:
                mask_patch = None

            if patch.shape[0] > 0 and patch.shape[1] > 0:
                patch = cv2.resize(patch, (self.patch_size, self.patch_size))
                if mask_patch is not None:
                    mask_patch = cv2.resize(mask_patch, (self.patch_size, self.patch_size))
        else:
            patch = np.zeros((self.patch_size, self.patch_size, image.shape[-1]))
            mask_patch = np.zeros((self.patch_size, self.patch_size))
        return patch, mask_patch

    def _print_class_distribution(self):
        class_counts = {}
        for label in self.label_post:
            class_counts[label] = class_counts.get(label, 0) + 1

        if 'test' not in self.split:
            class_names = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']
            total_count = []
            for i, name in enumerate(class_names):
                count = class_counts.get(i, 0)
                total_count.append(count)
                print(f'{name}: {count}')
        print('Número total de parches de edificios encontrados: '+str(len(self.label_post)))

    def __len__(self):
        return len(self.patch_post)

    def __getitem__(self, idx):
        # Images (pre-, post-) and mask (pre-)
        # print(self.patch_image_pre[idx])
        image_pre, image_post = self._process_image(idx)
        image_pre = image_pre / 255.0
        image_post = image_post / 255.0

        mask = self._create_mask(self.label_post_path[idx])

        patch_pre, _ = self._extract_patch(image_pre, mask, self.patch_post[idx])
        patch_post, mask_patch = self._extract_patch(image_post, mask, self.patch_post[idx])

        label_pre = self.label_pre[idx]
        label_post = self.label_post[idx]

        sample = {'patch_pre': patch_pre,
                  'patch_post': patch_post,
                  'mask_patch': mask_patch,
                  'label_pre': label_pre,
                  'label_post': label_post,
                  'idx': idx}
        if self.transform:
            sample = self.transform(sample)
        return sample


# In[4]:


patch_size = 64
train_dataset = xBDDataset('xBD_UC3M', ['train'], patch_size=patch_size)


# In[5]:


val_dataset = xBDDataset('xBD_UC3M', ['val'], patch_size=patch_size)


# In[6]:


# Elegimos un índice aleatorio
idx = np.random.randint(0, len(train_dataset))

# Obtenemos las imágenes completas originales usando la función interna
image_pre_full, image_post_full = train_dataset._process_image(idx)
event = train_dataset.label_post_path[idx].split('/')[-1][:-5]

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle(f"Imágenes completas (1024x1024) - Índice: {idx} - Evento: {event}", fontsize=16)

axes[0].imshow(image_pre_full)
axes[0].set_title("Antes del desastre (Pre)")
axes[0].axis('off')

axes[1].imshow(image_post_full)
axes[1].set_title("Después del desastre (Post)")
axes[1].axis('off')

plt.tight_layout()
plt.show()


# In[7]:


# Elegimos un índice aleatorio
idx = np.random.randint(0, len(train_dataset))

# Obtenemos un ejemplo procesado directamente desde el método __getitem__
sample = train_dataset[idx]

patch_pre = sample['patch_pre']
patch_post = sample['patch_post']
mask_patch = sample['mask_patch']
label_post = sample['label_post']

# Mapeo inverso de las clases de daño para la visualización
inverse_damage_classes = {v: k for k, v in train_dataset.damage_classes.items()}
damage_text = inverse_damage_classes[label_post]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"Parche de Edificio (64x64) - Etiqueta Post-Desastre: {damage_text} ({label_post})", fontsize=16)

# Aseguramos que los valores estén en el rango correcto para matplotlib (0-255 o 0-1)
axes[0].imshow(patch_pre)
axes[0].set_title("Parche Pre-desastre")
axes[0].axis('off')

axes[1].imshow(patch_post)
axes[1].set_title("Parche Post-desastre")
axes[1].axis('off')

axes[2].imshow(mask_patch, cmap='gray')
axes[2].set_title("Máscara del Edificio")
axes[2].axis('off')

plt.tight_layout()
plt.show()


# In[8]:


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
        image = sample['patch_post']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        if h>new_h:
            top = np.random.randint(0, h - new_h)
        else:
            top=0

        if w>new_w:
            left = np.random.randint(0, w - new_w)
        else:
            left = 0

        image = image[top: top + new_h,
                     left: left + new_w]

        sample = {'patch_pre': sample['patch_pre'],
                  'patch_post': image,
                  'mask_patch': sample['mask_patch'],
                  'label_pre': sample['label_pre'],
                  'label_post': sample['label_post'],
                  'idx': sample['idx']}
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
        image = sample['patch_post']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        rem_h = h - new_h
        rem_w = w - new_w

        if h>new_h:
            top = int(rem_h/2)
        else:
            top=0

        if w>new_w:
            left = int(rem_w/2)
        else:
            left = 0

        image = image[top: top + new_h,
                     left: left + new_w]

        sample = {'patch_pre': sample['patch_pre'],
                  'patch_post': image,
                  'mask_patch': sample['mask_patch'],
                  'label_pre': sample['label_pre'],
                  'label_post': sample['label_post'],
                  'idx': sample['idx']}
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
        image = sample['patch_post']

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

        sample = {'patch_pre': sample['patch_pre'],
                  'patch_post': image,
                  'mask_patch': sample['mask_patch'],
                  'label_pre': sample['label_pre'],
                  'label_post': sample['label_post'],
                  'idx': sample['idx']}
        return sample


class ToTensor(object):
    """Convertimos ndarrays de la muestra en tensores."""

    def __call__(self, sample):
        image, label = sample['patch_post'], sample['label_post']

        # Cambiamos los ejes
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image.astype(np.float32))

        label=torch.tensor(label,dtype=torch.long)

        sample = {'patch_pre': sample['patch_pre'],
                  'patch_post': image,
                  'mask_patch': sample['mask_patch'],
                  'label_pre': sample['label_pre'],
                  'label_post': label,
                  'idx': sample['idx']}
        return sample


class Normalize(object):
    """Normalizamos los datos restando la media y dividiendo por las desviaciones típicas.

    Args:
        mean_vec: El vector con las medias.
        std_vec: el vector con las desviaciones típicas.
    """

    def __init__(self, mean,std):

        assert len(mean)==len(std),'Length of mean and std vectors is not the same'
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, sample):
        image = sample['patch_post']
        c, h, w = image.shape
        assert c==len(self.mean), 'Length of mean and image is not the same'
        dtype = image.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=image.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=image.device)
        image.sub_(mean[:, None, None]).div_(std[:, None, None])

        sample = {'patch_pre': sample['patch_pre'],
                  'patch_post': image,
                  'mask_patch': sample['mask_patch'],
                  'label_pre': sample['label_pre'],
                  'label_post': sample['label_post'],
                  'idx': sample['idx']}
        return sample


# In[9]:


class AugmentPostPatch(object):
    """
    Extrae 'patch_post' del diccionario, le aplica transformaciones de torchvision 
    y lo devuelve al diccionario.
    """
    def __init__(self, transform_pipeline):
        self.transform_pipeline = transform_pipeline

    def __call__(self, sample):
        image = sample['patch_post']
        
        # 1. Convertimos la matriz numpy a imagen PIL (formato de 8 bits: 0-255)
        pil_image = Image.fromarray(util.img_as_ubyte(image))
        
        # 2. Aplicamos el pipeline de transformaciones (ej. flips, rotaciones)
        transformed_image = self.transform_pipeline(pil_image)
        
        # 3. Volvemos a convertir a matriz numpy flotante (0.0 a 1.0)
        image_np = util.img_as_float(np.asarray(transformed_image))
        
        # 4. Actualizamos el diccionario y lo devolvemos
        sample['patch_post'] = image_np
        return sample


# In[10]:


# Definimos las transformaciones visuales. 
vision_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5), # Voltea la imagen horizontalmente (50% de prob)
    transforms.RandomVerticalFlip(p=0.5),   # Voltea la imagen verticalmente (50% de prob)
    transforms.RandomRotation(degrees=45),  # Rota el edificio aleatoriamente hasta 45 grados
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05) # Simula cambios de iluminación/clima
])

# Componemos el pipeline completo para Train (Aumentación + Tensores + Normalización)
train_transforms = transforms.Compose([
    AugmentPostPatch(vision_transforms), # 1. Aplicamos data augmentation
    ToTensor(),                          # 2. Convertimos a tensor
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 3. Normalizamos
])

# Pipeline para Validación y Test sin augmentation
eval_transforms = transforms.Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Datos de train
train_dataset = xBDDataset('xBD_UC3M', ['train'], patch_size=64,
                           max_size=500, # set this to 0 when training
                           transform=train_transforms)

# Datos de validación
val_dataset = xBDDataset('xBD_UC3M', ['val'], patch_size=64,
                         transform=eval_transforms)

# Datos de test
test_dataset = xBDDataset('xBD_UC3M', ['test'], patch_size=64,
                          transform=eval_transforms)


# In[11]:


#Especificamos el dataset de train, un tamaño de batch de 64, desordenamos las imágenes,
# y paralelizamos con 3 workers
train_dataloader = DataLoader(train_dataset, batch_size=64,
                        shuffle=True, num_workers=3)

#Dataset de validación => No desordenamos
#Como no hay que hacer backward, podemos aumentar mucho el batch_size
val_dataloader = DataLoader(val_dataset, batch_size=128,
                        shuffle=False, num_workers=3)

#Dataset de test => No desordenamos
test_dataloader = DataLoader(test_dataset, batch_size=128,
                        shuffle=False, num_workers=3)




# In[12]:


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
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(16) # Batch Normalization accelerates training
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(16)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output shape after pool: [Batch_size, 16, 32, 32]
        
        # -------------------------------------------------------------------
        # BLOCK 2
        # Input shape:[Batch_size, 16, 32, 32]
        # -------------------------------------------------------------------
        self.conv2_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(32)
        self.conv2_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(32)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output shape after pool:[Batch_size, 32, 16, 16]

        # -------------------------------------------------------------------
        # BLOCK 3
        # Input shape: [Batch_size, 32, 16, 16]
        # -------------------------------------------------------------------
        self.conv3_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(64)
        self.conv3_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
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
        x = self.dropout(x)   # Apply dropout during training
        x = self.fc2(x)       # Produce the final 4 logits
        
        return x


# In[13]:


customNet = CustomNet() #Invocamos el constructor de la red (método init())
customNet.to(device) #Pasamos la red al device que estemos usando (gpu)
#Obtenemos un batch de datos y extraemos imágenes y etiquetas
data=next(iter(train_dataloader))
inputs = data['patch_post'].to(device).float()
labels = data['label_post'].to(device)

batchSize = labels.shape
print('El tamaño del tensor que representa un batch de imágenes es {}'.format(inputs.shape))

#Lo pasamos por la red
with torch.set_grad_enabled(False):
    outputs = customNet(inputs)
    print('El tamaño del tensor de salida es {}'.format(outputs.shape))


# In[14]:


criterion = nn.CrossEntropyLoss()

# Usaremos SGD con momento para optimizar
optimizer_ft = optim.SGD(customNet.parameters(), lr=1e-2, momentum=0.9)

# Un factor lr que  decae 0.1 cada 7 épocas
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# In[15]:


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, plot_confusion_matrix=True):

    #Fijamos semillas para maximizar reproducibilidad
    random.seed(42)
    npr.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False

    since = time.time()

    numClasses = len(list(image_datasets['train'].damage_classes.keys()))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0
    best_outputs = []
    best_labels = []
    #Bucle de épocas de entrenamiento
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Cada época tiene entrenamiento y validación
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Ponemos el modelo en modo entrenamiento
            else:
                model.eval()   # Ponemos el modelo en modo evaluación


            #Tamaño del dataset
            numSamples = dataset_sizes[phase]

            # Creamos las variables que almacenarán las salidas y las etiquetas
            outputs_m=np.zeros((numSamples,numClasses),dtype=np.float32)
            labels_m=np.zeros((numSamples,),dtype=int)
            running_loss = 0.0

            contSamples=0

            # Iteramos sobre los datos.
            for sample in dataloaders[phase]:
                inputs = sample['patch_post'].to(device)
                labels = sample['label_post'].to(device)

                #Tamaño del batch
                batchSize = labels.shape[0]

                # Ponemos a cero los gradientes
                optimizer.zero_grad()

                # Paso forward
                # registramos operaciones solo en train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward y optimización solo en training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Sacamos estadísticas y actualizamos variables
                running_loss += loss.item() * inputs.size(0)

                #Aplicamos un softmax a la salida
                outputs=F.softmax(outputs.data,dim=1)
                outputs_m [contSamples:contSamples+batchSize,...]=outputs.cpu().numpy()
                labels_m [contSamples:contSamples+batchSize]=labels.cpu().numpy()
                contSamples+=batchSize

            #Actualizamos la estrategia de lr
            if phase == 'train':
                scheduler.step()

            #Loss acumulada en la época
            epoch_loss = running_loss / dataset_sizes[phase]

            #Calculamos Macro F1-score
            epoch_f1=f1_score(labels_m,np.argmax(outputs_m,axis=1),average='macro')

            print('{} Loss: {:.4f} Macro F1-score: {:.4f}'.format(
                phase, epoch_loss, epoch_f1))

            # copia profunda del mejor modelo
            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())
                best_outputs = np.argmax(outputs_m,axis=1)
                best_labels = labels_m

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Macro F1-score: {:4f}'.format(best_f1))

    if plot_confusion_matrix:
        # Visualizamos la matriz de confusión
        cm=confusion_matrix(best_labels, best_outputs, normalize='true')
        #print(cm)
        #Vamos a mostrar en porcentajes en vez de probs
        ncmd=ConfusionMatrixDisplay(100*cm,display_labels=list(image_datasets['val'].damage_classes.keys()))
        ncmd.plot(xticks_rotation='vertical')
        plt.title('Normalized confusion matrix (%)')
        plt.show()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# In[16]:


image_datasets = {'train' : train_dataset, 'val': val_dataset, 'test': test_dataset}

dataloaders = {'train' : train_dataloader, 'val': val_dataloader, 'test': test_dataloader}

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}
class_names = list(image_datasets['train'].damage_classes.keys())


# In[19]:


from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[ ]:


customNet = train_model(customNet, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=20)


# In[ ]:




