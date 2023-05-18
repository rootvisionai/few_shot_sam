import base64
import io
from PIL import Image
import yaml
import json
import types
import numpy as np


def get_image(image_data):
    # base64 encoded string
    image_data = base64.b64decode(image_data)
    image_ = Image.open(io.BytesIO(image_data))
    image = Image.new("RGB", image_.size)
    image.paste(image_)
    image = np.asarray(image)
    return image

def load_config(path_to_config_yaml="./config.yaml"):
    with open(path_to_config_yaml) as f:
        dct = yaml.safe_load(f)

    def load_object(dct):
        return types.SimpleNamespace(**dct)

    cfg = json.loads(json.dumps(dct), object_hook=load_object)
    return cfg

def adapt_point(pts, initial_shape, final_shape):
    scale_y = final_shape[0] / initial_shape[0]
    scale_x = final_shape[1] / initial_shape[1]
    pts_ = {}
    pts_["y"] = pts["y"] * scale_y
    pts_["x"] = pts["x"] * scale_x
    return pts_

def get_embedding(predictor, image, point):
    predictor.set_image(image)
    features = predictor.features
    point_ = {"y": int(point[1]), "x": int(point[0])}
    point_adapted = adapt_point(point_, initial_shape=image.shape[0:2], final_shape=features.shape[-2:])
    embedding = features[:, :, int(point_adapted["y"]), int(point_adapted["x"])]
    return embedding

def merge_multilabel_masks(masks, labels_int, COLORMAP):
    masks = np.transpose(np.array(masks), (1, 2, 0))
    height, width = masks.shape[0:2]
    rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(masks.shape[-1]):
        coords = np.where(masks[..., i] == 1)
        rgb_mask[coords[0], coords[1], :] = tuple(COLORMAP[labels_int[i]])
    return rgb_mask

def get_logger(log_path = 'logs/file.log'):
    import logging

    # Create a custom logger
    logger = logging.getLogger(__name__)

    # Set level of logger
    logger.setLevel(logging.DEBUG)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_path)

    # Set level of handlers
    c_handler.setLevel(logging.WARNING)
    f_handler.setLevel(logging.ERROR)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger
