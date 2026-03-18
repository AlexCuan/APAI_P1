import os
import json
import cv2
import tifffile
import numpy as np
from shapely.wkt import loads
from shapely.geometry import Polygon
from natsort import natsorted
from tqdm import tqdm

DATA_DIR = './xBD_UC3M'
OUTPUT_DIR = './xBD_cropped'
PATCH_SIZE = 64

damage_classes = {
    'no-damage': 0,
    'minor-damage': 1,
    'major-damage': 2,
    'destroyed': 3
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

for split in ['train', 'val', 'test']:
    print(f"Procesando split: {split}")
    split_out_dir = os.path.join(OUTPUT_DIR, split)
    os.makedirs(split_out_dir, exist_ok=True)

    disaster_dirs = os.listdir(os.path.join(DATA_DIR, split))

    image_post_files = []
    label_post_files = []

    # Recorremos carpetas respetando la estructura
    for d in disaster_dirs:
        disaster_path = os.path.join(DATA_DIR, split, d)
        if not os.path.isdir(disaster_path): continue

        image_files = os.listdir(os.path.join(disaster_path, 'images'))
        labels_files = os.listdir(os.path.join(disaster_path, 'labels'))

        image_post_files.append(
            natsorted([os.path.join(disaster_path, 'images', f) for f in image_files if 'post' in f]))
        label_post_files.append(
            natsorted([os.path.join(disaster_path, 'labels', f) for f in labels_files if 'post' in f]))

    image_post_files = natsorted(np.hstack(image_post_files))
    label_post_files = natsorted(np.hstack(label_post_files))

    global_idx = 0

    # Extraemos parches imagen a imagen
    for i in tqdm(range(len(image_post_files)), desc=f"Recortando {split}"):
        img_path = image_post_files[i]
        lbl_path = label_post_files[i]

        image_post = None

        with open(lbl_path, 'r') as f:
            data = json.load(f)

        if 'features' in data and 'xy' in data['features']:
            features = data['features']['xy']
            for feature in features:
                if 'wkt' in feature and 'properties' in feature:
                    props = feature['properties']
                    if props.get('feature_type') == 'building':
                        damage_type = props.get('subtype', 'no-damage')

                        # Condición original (incluye el -1 o 'un-classified' de test)
                        if damage_type in damage_classes or damage_type == -1 or damage_type == 'un-classified':
                            label = damage_classes.get(damage_type, -1)

                            # Cargado Perezoso: Sólo lee el TIFF si hay al menos 1 polígono válido
                            if image_post is None:
                                image_post = tifffile.imread(img_path)

                                # --- CORRECCIÓN: Convertir 16-bit a 8-bit (uint8) ---
                                image_post = np.clip(image_post, 0, 255).astype(np.uint8)
                                # ----------------------------------------------------

                                if image_post.shape[:2] != (1024, 1024):
                                    image_post = cv2.resize(image_post, (1024, 1024))

                            geom = loads(feature['wkt'])
                            if isinstance(geom, Polygon) and geom.is_valid:
                                coords = np.array(geom.exterior.coords, dtype=np.int32)
                                x_min, y_min = coords.min(axis=0)
                                x_max, y_max = coords.max(axis=0)
                                center_x = (x_min + x_max) // 2
                                center_y = (y_min + y_max) // 2
                                half_size = PATCH_SIZE // 2

                                x1 = max(0, center_x - half_size)
                                y1 = max(0, center_y - half_size)
                                x2 = min(image_post.shape[1], center_x + half_size)
                                y2 = min(image_post.shape[0], center_y + half_size)

                                patch = image_post[y1:y2, x1:x2]
                                if patch.shape[0] > 0 and patch.shape[1] > 0:
                                    patch = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE))
                                else:
                                    patch = np.zeros((PATCH_SIZE, PATCH_SIZE, image_post.shape[-1]),
                                                     dtype=image_post.dtype)
                            else:
                                patch = np.zeros((PATCH_SIZE, PATCH_SIZE, image_post.shape[-1]), dtype=image_post.dtype)

                            # Formato Nombre: {id_global}_{etiqueta}.png
                            out_filename = f"{global_idx:06d}_{label}.png"
                            out_filepath = os.path.join(split_out_dir, out_filename)

                            # Convertir de RGB (TIF) a BGR (Opencv) antes de escribir
                            patch_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR) if patch.shape[-1] == 3 else patch
                            cv2.imwrite(out_filepath, patch_bgr)

                            global_idx += 1

print("\n¡Extracción finalizada! Ya puedes usar CroppedxBDDataset.")