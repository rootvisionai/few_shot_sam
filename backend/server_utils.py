import base64
import io
from PIL import Image
import yaml
import json
import types
import numpy as np
import torch
import logging
import os
import requests
import tqdm


def get_image(image_data):
    # base64 encoded string
    image_data = base64.b64decode(image_data)
    image_ = Image.open(io.BytesIO(image_data))
    image = Image.new("RGB", image_.size)
    image.paste(image_)
    image = np.asarray(image)
    return image

def initialize_model(file_path):
    if not os.path.isdir("./checkpoints"):
        os.makedirs("./checkpoints")

    if not os.path.isfile(file_path):
        checkpoint_split = file_path.split("/")[-1]
        url = 'https://dl.fbaipublicfiles.com/segment_anything/'+checkpoint_split  # The URL of the file you want to download
        response = requests.get(url, stream=True)

        # Get the size of the content (in bytes)
        total_size = int(response.headers.get('content-length', 0))

        # Block size to be displayed during the download process (1 Kilobyte)
        block_size = 1024

        with open(file_path, 'wb') as file:
            for data in tqdm.tqdm(response.iter_content(block_size), total=total_size//block_size, unit='KB'):
                file.write(data)

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

def l2_norm(vector):
    v_norm = vector.norm(dim=-1, p=2)
    vector = vector.divide(v_norm.unsqueeze(-1))
    return vector

def flatten_feature_map(embeddings):
    """
    :param embeddings: shape B, N, H, W
    :return: B, H*W, N
    """
    b, n, h, w = embeddings.shape
    embeddings = embeddings.reshape(b, n, h*w).permute(0, 2, 1)
    return embeddings

def get_similarity(support_emb, query_features):

    support_emb = l2_norm(support_emb)

    b, n, h, w = query_features.shape
    query_features = flatten_feature_map(query_features)
    query_features = l2_norm(query_features)[0]

    cos_sim = torch.nn.functional.linear(
        support_emb.float(),
        query_features.float()
    )

    cos_sim = cos_sim.reshape(b, h, w).squeeze(0)
    return cos_sim

def get_embedding(predictor, image, point):
    with torch.no_grad():
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

def numpy_to_base64(image: np.ndarray) -> str:
    # Ensure the image array is an 8-bit integer
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Convert numpy array to PIL image
    pil_image = Image.fromarray(image)

    # Create a BytesIO object and save the image
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format='JPEG')  # use appropriate format based on your needs

    # Encode BytesIO as base64 and return it
    base64_encoded_image = base64.b64encode(byte_arr.getvalue()).decode('ascii')  # decode to create a string

    return base64_encoded_image