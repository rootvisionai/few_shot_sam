import yaml
import json
import types
import torch
import numpy as np
from PIL import Image
# from kmeans_pytorch import kmeans, kmeans_predict


def load_config(path_to_config_yaml="./config.yaml"):

    with open(path_to_config_yaml) as f:
        dct = yaml.safe_load(f)

    def load_object(dct):
        return types.SimpleNamespace(**dct)

    cfg = json.loads(json.dumps(dct), object_hook=load_object)

    return cfg

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
        support_emb,
        query_features
    )

    cos_sim = cos_sim.reshape(b, h, w).squeeze(0)
    return cos_sim

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

# def cluster_embeddings(feature_map, num_clusters, device="cpu"):
#     embeddings = flatten_feature_map(feature_map)
#     cluster_ids, cluster_centers = kmeans(
#         X=embeddings,
#         num_clusters=num_clusters,
#         distance='cosine',
#         device=torch.device(device)
#     )
#     return cluster_ids, cluster_centers
#
# def get_clusteer_center(cluster_centers, embedding, device="cpu"):
#     cluster_ids = kmeans_predict(
#         embedding,
#         cluster_centers,
#         'cosine',
#         device=device
#     )
#     predicted_cluster_center = cluster_centers[cluster_ids]
#     return predicted_cluster_center

def import_image(path):
    image_ = Image.open(path)
    image = Image.new("RGB", image_.size)
    image.paste(image_)
    image = np.asarray(image)
    return image
