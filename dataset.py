"""YOLO polygons -> original-resolution masks -> nearest-neighbor resize."""
from pathlib import Path
import hashlib
import json
import random
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from common import image_tensor


def class_names(root):
    config = yaml.safe_load((Path(root)/'data_seg.yaml').read_text(encoding='utf-8'))
    names = config['names']
    return {int(k): str(v) for k, v in (names.items() if isinstance(names, dict)
                                       else enumerate(names))}


def manifest(root, split):
    root = Path(root).resolve()
    file = root/(split if str(split).endswith('.txt') else f'{split}.txt')
    rows = [r.strip().replace('\\', '/') for r in file.read_text(encoding='utf-8').splitlines()
            if r.strip()]
    paths = [Path(r) if Path(r).is_absolute() else root/r for r in rows]
    if len(set(paths)) != len(paths):
        raise ValueError(f'Duplicate entries in {file}')
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)
    return paths


def label_path(root, image):
    relative = Path(image).resolve().relative_to((Path(root)/'images').resolve())
    return Path(root)/'labels'/relative.with_suffix('.txt')


def polygons(path, names):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing annotation: {path}; not silently treated as empty')
    result = []
    for index, row in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not row.strip():
            continue
        vals = np.asarray([float(x) for x in row.split()], dtype=np.float64)
        if not np.isfinite(vals).all() or len(vals) < 7 or len(vals) % 2 != 1:
            raise ValueError(f'{path}:{index}: malformed polygon')
        cls = int(vals[0])
        if cls != vals[0] or cls not in names:
            raise ValueError(f'{path}:{index}: invalid class')
        points = vals[1:].reshape(-1, 2)
        if (points < 0).any() or (points > 1).any() or len(np.unique(points, axis=0)) < 3:
            raise ValueError(f'{path}:{index}: invalid normalized coordinates')
        area = abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) -
                   np.dot(points[:, 1], np.roll(points[:, 0], 1)))/2
        if area <= 1e-12:
            raise ValueError(f'{path}:{index}: zero-area polygon')
        result.append((cls, points))
    return result


def lane_mask(root, image, size, lane_class=3, names=None):
    names = names or class_names(root)
    if names.get(lane_class) != 'lane':
        raise ValueError('Selected class must be named lane in data_seg.yaml')
    width, height = size
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    for cls, points in polygons(label_path(root, image), names):
        if cls == lane_class:
            xy = [(min(width-1, max(0, round(x*width))),
                   min(height-1, max(0, round(y*height)))) for x, y in points]
            draw.polygon(xy, fill=255)
    return mask


def manifest_hash(root, split):
    file = Path(root)/(split if str(split).endswith('.txt') else f'{split}.txt')
    return hashlib.sha256(file.read_bytes()).hexdigest()


class LaneDataset(Dataset):
    def __init__(self, root, split, height=36, width=64, augment=False, lane_class=3):
        self.paths = manifest(root, split)
        self.augment = augment
        names = class_names(root)
        self.images, self.masks = [], []
        # Cache resized samples only (~36 MB total), not all full-resolution images.
        for path in self.paths:
            with Image.open(path) as im:
                mask = lane_mask(root, path, im.size, lane_class, names)
                self.images.append(image_tensor(im, height, width))
                array = np.array(mask.resize((width, height), Image.Resampling.NEAREST),
                                 dtype=np.float32)/255.0
                self.masks.append(torch.from_numpy(array[None].copy()))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = self.images[i].clone()
        mask = self.masks[i]
        if self.augment:
            if random.random() < 0.5:
                image = image*torch.tensor([random.uniform(.95, 1.05) for _ in range(3)])[:, None, None]
            if random.random() < 0.5:
                image = image*random.uniform(.9, 1.1)
            image = image.clamp(0, 1)
            if random.random() < 0.2:
                image = TF.gaussian_blur(image, [3, 3], [random.uniform(.2, .7)]*2)
        return image, mask
