import json
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from model import CustomUNet

ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_SPLIT = '../../splits/65-35/train.txt'
DEFAULT_VAL_SPLIT = '../../splits/65-35/val.txt'


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding='utf-8')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_for(name='auto'):
    return torch.device('cuda' if name == 'auto' and torch.cuda.is_available()
                        else 'cpu' if name == 'auto' else name)


def load_model(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = CustomUNet(checkpoint['config']['base_channels'])
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval(), checkpoint


def image_tensor(image, height, width):
    image = image.convert('RGB').resize((width, height), Image.Resampling.BILINEAR)
    arr = np.array(image, dtype=np.float32)/255.0
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


def iou_arrays(prediction, truth):
    p, t = prediction.astype(bool), truth.astype(bool)
    intersection = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    return float(intersection/union) if union else 1.0
