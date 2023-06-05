import torch
from flask import Flask, request, jsonify
from waitress import serve
import numpy as np
import os
import sys
import cv2
import json
import time
import queue
# import threading

import server_utils as utils
import annotations
from exact_solution import ExactSolution
# from forwarder import Forwarder

from segment_anything import sam_model_registry, SamPredictor


model_to_checkpoint_map = {
    "vit_b": "./checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": "./checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": "./checkpoints/sam_vit_h_4b8939.pth",
}
base = "localhost:8080"
ExtractInQueue = queue.Queue(maxsize=10)
ExtractOutDict = {}
GenerateInQueue = queue.Queue(maxsize=4)
GenerateOutDict = {}
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    return json.dumps({"status": "active"})

@app.route('/extract_features', methods=['POST'])
def extract():

    """

    input: {
        "images": [base64_encoded_image, ...],
        "annotations": {
            [
                {"coordinates": {"positive": [x, y], "negative": [x, y]}, "label": "label_name1", image_id: "0", "image_path"},
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
        for i, _ in enumerate(data['annotations']):
            annotations = data['annotations'][i]
            if not all([elm in annotations for elm in ["coordinates", "label", "image_id"]]):
                continue

            coordinates = annotations["coordinates"]
            label = annotations["label"]
            image_id = int(annotations["image_id"])
            image = utils.get_image(data["images"][image_id])

            positive_coord = coordinates["positive"]
            negative_coord = coordinates["negative"]

            t0 = time.time()
            if not None in positive_coord:
                for pt in positive_coord:
                    embedding = utils.get_embedding(predictor, image, pt)
                    embeddings.append(embedding.cpu().numpy().tolist())
            t1 = time.time()
            print(f"POSITIVE POINT INFERENCE TIME: {t1-t0}")

            t0 = time.time()
            if not None in negative_coord:
                for pt in negative_coord:
                    n_embedding = utils.get_embedding(predictor, image, pt)
                    n_embeddings.append(n_embedding.cpu().numpy().tolist())
            t1 = time.time()
            print(f"NEGATIVE POINT INFERENCE TIME: {t1 - t0}")

            if label not in support_package:
                support_package[label] = {"positive": [], "negative": []}

            support_package[label]["positive"].append(embeddings)
            support_package[label]["negative"].append(n_embeddings)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        error_text = f"{exc_type}\n-file {fname}\n--line {exc_tb.tb_lineno}\n{e}"
        logger.error(error_text)

    response = {'package': support_package, "error": "error_text"}
    return json.dumps(response)

@app.route('/generate/<gen_type>', methods=['POST'])
def generate(gen_type):
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
            image_path = request.json['image_path']
            for label_int, label_str in enumerate(support_package):
                image = utils.get_image(image_data)
                q_image_shape = image.shape
                t0 = time.time()
                predictor.set_image(image)
                features = predictor.features
                t1 = time.time()
                print(f"SAM INFERENCE TIME: {t1-t0}")
                linear_model_labels_int = []
                embedding_collection = []
                for i, embedding in enumerate(support_package[label_str]["positive"][0]):
                    embedding = np.array(embedding)
                    embedding = torch.from_numpy(embedding).to(cfg.device)[0]

                    linear_model_labels_int.append(1)
                    embedding_collection.append(embedding)

                for i, embedding in enumerate(support_package[label_str]["negative"][0]):
                    embedding = np.array(embedding)
                    embedding = torch.from_numpy(embedding).to(cfg.device)[0]

                    linear_model_labels_int.append(0)
                    embedding_collection.append(embedding)

                embedding_collection = torch.stack(embedding_collection, dim=0)
                linear_model_labels_int = np.array(linear_model_labels_int)

                t0 = time.time()
                linear_model = ExactSolution(
                    device=cfg.device,
                    embedding_collection=embedding_collection,
                    labels_int=linear_model_labels_int,
                    threshold=cfg.threshold
                )
                predictions = linear_model.infer(features)
                t1 = time.time()
                print(f"EXACT SOLUTION INFERENCE TIME: {t1 - t0}")

                yx_multi = (predictions == 1.).nonzero()  # this can be changed later

                for yx in yx_multi:
                    t0 = time.time()
                    xy = utils.adapt_point(
                        {"x": yx[1].item(), "y": yx[0].item()},
                        initial_shape=features.shape[-2:],
                        final_shape=image.shape[0:2]
                    )
                    t1 = time.time()
                    print(f"ADAPT POINT TIME: {t1 - t0}")

                    matching_points[label_str] = {"x": xy["x"], "y": xy["y"]}
                    if gen_type == "point":
                        return jsonify({'matching_points': matching_points, "error": error_text})

                    l_ = np.ones((1,))
                    t0 = time.time()
                    mask_, scores, logits = predictor.predict(
                        point_coords=np.array([[xy["x"], xy["y"]]]).astype(int),
                        point_labels=l_,
                        multimask_output=True,
                    )
                    t1 = time.time()
                    print(f"MASK GEN INFERENCE TIME: {t1 - t0}")
                    mask_ = mask_.astype(np.uint8)
                    mask_ = cv2.resize(mask_[0], (q_image_shape[1], q_image_shape[0]))

                    polygons, points_ = annotations.generate_polygons_from_mask(
                        polygons=polygons,
                        mask=mask_,
                        label=label_str,
                        polygon_resolution=cfg.labeling.polygon_resolution
                    )

                    masks.append(mask_)

                    t0 = time.time()
                    # create xml file from the coordinates
                    if len(points_)>0:
                        for cnt_p, pts_ in enumerate(points_):
                            # coordinates = np.nonzero(mask_)
                            if len(pts_) >= 3:
                                print(f"---> Finding bounding box: {cnt_p}/{len(points_)}")
                                pts = np.array([np.array(pt) for pt in pts_])
                                y0, y1, x0, x1 = pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()
                                bboxes.append({
                                    "coordinates": [int(x0), int(y0), int(x1), int(y1)],
                                    "format": "xyxy",
                                    "label": "label_str"
                                })

                                labels_str.append(label_str)
                                labels_int.append(label_int)

                    t1 = time.time()
                    print(f"BBOX GENERATION TIME: {t1 - t0}")


            if len(masks) > 0:
                print("---> Merging masks")
                masks_transformed = utils.merge_multilabel_masks(masks, labels_int, COLORMAP=cfg.COLORMAP)
                masks_transformed = masks_transformed * 255
                masks_encoded = utils.numpy_to_base64(masks_transformed)

                print("---> Creating bounding box annotations")
                pascal_xml = annotations.create_xml_multilabel(image_name=image_path, labels=labels_str, bboxes=bboxes)

                print("---> Creating polygon annotations")
                coco_json = annotations.create_polygon_json(image_path=image_path, image_data=image_data, polygons=polygons, size=masks_transformed.shape)

                if gen_type == "annotation":
                    response = {
                            "polygons": polygons,
                            "bounding_boxes": bboxes,
                            "coco_json": coco_json,
                            "pascal_xml": pascal_xml,
                            "error": error_text.replace("\n", " ")
                        }
                    return json.dumps(response)

                elif gen_type == "all":
                    response = {
                            "matching_points": matching_points,
                            "masks": masks_encoded,
                            "polygons": polygons,
                            "bounding_boxes": bboxes,
                            "coco_json": coco_json,
                            "pascal_xml": pascal_xml,
                            "error": error_text.replace("\n", " ")
                        }
                    return json.dumps(response)

            else:
                error_text = "NO MASK IS GENERATED FOR THIS IMAGE BASED ON THE GIVEN COORDINATES."
                logger.error(error_text)

        else:
            error_text = f"Given generate/{gen_type} is wrong."
            logger.error(error_text)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        error_text = f"{exc_type}\n-file {fname}\n--line {exc_tb.tb_lineno}\n{e}"
        logger.error(error_text)

    return jsonify({"error": error_text})

# @app.route('/forwarder/extract_features', methods=['POST'])
# def forwarder_extract():
#
#     timestamp = time.time()
#     if not ExtractInQueue.full():
#         ExtractInQueue.put({
#             "url": f"http://{base}/extract_features",
#             "data": request.json,
#             "timestamp": timestamp
#         })
#     else:
#         return json.dumps({"warning": "Queue is full, please wait and try again."})
#
#     while True:
#         if timestamp in ExtractOutDict:
#             response = ExtractOutDict[timestamp]["data"]
#             del ExtractOutDict[timestamp]
#             break
#
#     return response
#
# @app.route('/forwarder/generate/<gen_type>', methods=['POST'])
# def forwarder_generate(gen_type):
#
#     timestamp = time.time()
#     if not GenerateInQueue.full():
#         GenerateInQueue.put({
#             "url": f"http://{base}/generate/{gen_type}",
#             "data": request.json,
#             "timestamp": timestamp
#         })
#     else:
#         return json.dumps({"warning": "Queue is full, please wait and try again."})
#
#     while True:
#         if timestamp in GenerateOutDict:
#             response = GenerateOutDict[timestamp]["data"]
#             del GenerateOutDict[timestamp]
#             break
#
#     return response

def run_forwarder(forwarder):
    forwarder.run()


if __name__ == '__main__':
    cfg = utils.load_config("./config.yml")

    checkpoint = model_to_checkpoint_map[cfg.model]
    utils.initialize_model(checkpoint)

    sam = sam_model_registry[cfg.model](checkpoint=checkpoint)
    sam.to(device=cfg.device)
    predictor = SamPredictor(sam)
    
    if not os.path.isdir("./backend/logs/"):
        os.makedirs("./backend/logs/")
    
    if not os.path.isfile("./backend/logs/file.log"):
        with open("./backend/logs/file.log", "w") as fp:
            fp.write("")
    
    logger = utils.get_logger(log_path='./backend/logs/file.log')

    # forwarder_extract = Forwarder(
    #     in_queue=ExtractInQueue,
    #     out_dict=ExtractOutDict,
    #     freq=10
    # )
    #
    # forwarder_generate = Forwarder(
    #     in_queue=GenerateInQueue,
    #     out_dict=GenerateOutDict,
    #     freq=10
    # )
    #
    # fps = []
    # fps.append(threading.Thread(target=run_forwarder, args=(forwarder_extract,)))
    # fps.append(threading.Thread(target=run_forwarder, args=(forwarder_generate,)))
    #
    # for fp in fps:
    #     fp.daemon = True
    #     fp.start()

    # app.run(host = "0.0.0.0", port=8080, debug=False)
    serve(app, host="0.0.0.0", port=8080)
