import yaml
import json
import types
import torch


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


