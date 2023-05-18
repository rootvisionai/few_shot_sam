import torch
from flask import Flask, request, jsonify
import numpy as np
import os
import sys
import cv2

import server_utils as utils
import annotations

from segment_anything import sam_model_registry, SamPredictor
model_to_checkpoint_map = {
    "vit_b": "./checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": "./checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": "./checkpoints/sam_vit_h_4b8939.pth",
}

app = Flask(__name__)

@app.route('/extract_features', methods=['POST'])
def extract_features():

    """
    input: {
        "image": [base64_encoded_image, ...],
        "annotations": {
            [
                {"coordinates": [x, y], "label": "label_name"},
                ...
            ]
        }
    }
    output:
    "package": {
        "label_name": {
            "positive": [numpy array, ...],
            "negative": [numpy array, ...],
        },
        ...
    },
    "error": ...
    """

    try:
        data = request.json
        support_package = {}

        embeddings = []
        n_embeddings = []
        for i, image_data in enumerate(data['image']):
            image = utils.get_image(image_data)
            annotations = request.json['annotations'][i]

            for j, annot in enumerate(annotations):
                coordinates = annot["coordinates"]
                label = annot["label"]

                positive_coord = coordinates["positive"]
                negative_coord = coordinates["negative"]

                if not None in positive_coord:
                    for pt in positive_coord:
                        embedding = utils.get_embedding(predictor, image, pt)
                        embeddings.append(embedding)

                if not None in negative_coord:
                    for pt in negative_coord:
                        n_embedding = utils.get_embedding(predictor, image, pt)
                        n_embeddings.append(n_embedding)

                if label not in support_package:
                    support_package[label] = {"positive": [], "negative": []}

                support_package[label]["positive"].append(embeddings.cpu().numpy())
                support_package[label]["negative"].append(n_embeddings.cpu().numpy())

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        error_text = f"{exc_type}\n\t{fname}\n\t\t{exc_tb.tb_lineno}\n{e}"
        logger.error(error_text)

    return jsonify({'package': support_package, "error": error_text})

@app.route('/generate/<gen_type>', methods=['POST'])
def generate_mask(gen_type):
    """
    input: numpy embedding vectors
    output: binary mask
    """
    matching_points = {}

    polygons = []
    bboxes = []
    masks = []
    labels_str = []
    labels_int = []
    error_text = ""
    try:
        if gen_type in ["point", "annotation", "all"]:
            support_package = request.json['package']
            image_data = request.json['image']
            for label_int, label_str in enumerate(support_package):
                image = utils.get_image(image_data)
                q_image_shape = image.shape
                predictor.set_image(image)
                features = predictor.features

                similarity_maps = []
                for i, embedding in enumerate(support_package["positive"][label_str]):
                    embedding = torch.from_numpy(embedding).to(cfg.device)

                    similarity_map = utils.get_similarity(embedding, features)
                    similarity_map = torch.where(similarity_map > cfg.threshold, 1., 0.)
                    similarity_maps.append(similarity_map)

                for i, n_embedding in enumerate(support_package["negative"][label_str]):
                    embedding = torch.from_numpy(embedding).to(cfg.device)

                    similarity_map = utils.get_similarity(n_embedding, features)
                    similarity_map = torch.where(similarity_map > cfg.threshold, 1., 0.)
                    similarity_map = 1 - similarity_map
                    similarity_maps.append(similarity_map)

                similarity_maps = torch.stack(similarity_maps, dim=0)
                for i, sm_ in enumerate(similarity_maps):
                    if i == 0:
                        sm = sm_
                    else:
                        sm = sm * sm_
                similarity_maps = sm

                print(f"HIGHEST SIMILARITY: {torch.max(similarity_maps).item()}")
                # similarity_maps = torch.einsum('bij->ij', similarity_maps)

                yx_multi = (similarity_maps == torch.max(similarity_maps)).nonzero()  # this can be changed later

                for yx in yx_multi:
                    xy = utils.adapt_point(
                        {"x": yx[0, 1].item(), "y": yx[0, 0].item()},
                        initial_shape=features.shape[-2:],
                        final_shape=image.shape[0:2]
                    )

                    if gen_type == "point":
                        matching_points[label_str] = {"x": xy["x"], "y": xy["y"]}
                        return jsonify({'matching_points': matching_points, "error": error_text})

                    l_ = np.ones((1,))
                    mask_, scores, logits = predictor.predict(
                        point_coords=np.array([[xy["x"], xy["y"]]]).astype(int),
                        point_labels=l_,
                        multimask_output=True,
                    )
                    mask_ = mask_.astype(np.uint8)
                    mask_ = cv2.resize(mask_[0], (q_image_shape[1], q_image_shape[0]))

                    polygons = annotations.generate_polygons_from_mask(
                        polygons=polygons,
                        mask=mask_,
                        label=label_str,
                        polygon_resolution=cfg.labeling.polygon_resolution
                    )

                    masks.append(mask_)

                    # create xml file from the coordinates
                    coordinates = np.nonzero(mask_)
                    y0, y1, x0, x1 = coordinates[0].min(), coordinates[0].max(), coordinates[1].min(), coordinates[1].max()
                    bboxes.append([x0, y0, x1, y1])

                    labels_str.append(label_str)
                    labels_int.append(label_int)

            if len(masks) > 0:
                masks_transformed = utils.merge_multilabel_masks(masks, labels_int, COLORMAP=cfg.COLORMAP)
                masks_transformed = masks_transformed * 255

                pascal_xml = annotations.create_xml_multilabel(image_name=qip, labels=labels_str, bboxes=bboxes)
                coco_json = annotations.create_polygon_json(image_path=qip, polygons=polygons, size=masks_transformed.shape)

                if gen_type == "annotation":
                    return jsonify({
                        "polygons": polygons,
                        "bounding_boxes": bboxes,
                        "coco_json": coco_json,
                        "pascal_xml": pascal_xml,
                        "error": error_text
                    })

                elif gen_type == "all":
                    return jsonify({
                        'matching_points': matching_points,
                        "polygons": polygons,
                        "bounding_boxes": bboxes,
                        "coco_json": coco_json,
                        "pascal_xml": pascal_xml,
                        "error": error_text
                    })

            else:
                error_text = "NO MASK IS GENERATED FOR THIS IMAGE BASED ON THE GIVEN COORDINATES."
                logger.error(error_text)

        else:
            error_text = f"Given generate/{gen_type} is wrong."
            logger.error(error_text)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        error_text = f"{exc_type}\n\t{fname}\n\t\t{exc_tb.tb_lineno}\n{e}"
        logger.error(error_text)

    return jsonify({"error": error_text})


if __name__ == '__main__':
    cfg = utils.load_config("../config.yml")

    checkpoint = model_to_checkpoint_map[cfg.model]
    sam = sam_model_registry[cfg.model](checkpoint=checkpoint)
    sam.to(device=cfg.device)
    predictor = SamPredictor(sam)

    logger = utils.get_logger(log_path='logs/file.log')
    app.run(debug=True)