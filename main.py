import glob

import numpy as np
import torch
import os
import cv2
import torchvision.utils
from skimage.feature import peak_local_max

from segment_anything import sam_model_registry, SamPredictor
import interface
import annotations
import utils
from statistics import median

model_to_checkpoint_map = {
    "vit_b": "./checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": "./checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": "./checkpoints/sam_vit_h_4b8939.pth",
}


if __name__=="__main__":
    cfg = utils.load_config("./config.yml")

    checkpoint = model_to_checkpoint_map[cfg.model]

    sam = sam_model_registry[cfg.model](checkpoint=checkpoint)
    sam.to(device=cfg.device)

    predictor = SamPredictor(sam)
    support_package = {"positive": {}, "negative": {}}
    while True:
        # Support Images: Load, point, extract
        s_image_paths = glob.glob(os.path.join(cfg.data.support_dir, f"*.{cfg.data.format}"))
        embeddings = []
        n_embeddings = []
        for sip in s_image_paths:
            image = utils.import_image(sip)
            image = cv2.resize(image, (cfg.window_size[0], cfg.window_size[1]))

            points = interface.click_on_point(img=image)
            if not None in points:
                for pt in points:
                    embedding = utils.get_embedding(predictor, image, pt)
                    embeddings.append(embedding)

            points = interface.click_on_point(img=image)
            if not None in points:
                for pt in points:
                    n_embedding = utils.get_embedding(predictor, image, pt)
                    n_embeddings.append(n_embedding)

        label, event = interface.relabel_or_continue()
        support_package["positive"][label] = embeddings
        support_package["negative"][label] = n_embeddings
        if event == "END":
            break
        elif event == "CANCEL":
            break
        else:
            pass

    # Query Images: Load, extract, match
    q_image_paths = glob.glob(os.path.join(cfg.data.query_dir, f"*.{cfg.data.format}"))
    q_image_paths += glob.glob(os.path.join(cfg.data.query_dir, "**", f"*.{cfg.data.format}"))
    q_image_paths += glob.glob(os.path.join(cfg.data.query_dir, "**", "**", f"*.{cfg.data.format}"))
    q_image_paths += glob.glob(os.path.join(cfg.data.query_dir, "**", "**", "**", f"*.{cfg.data.format}"))
    for cnt, qip in enumerate(q_image_paths):
        masks = []
        bboxes = []
        labels_str = []
        labels_int = []
        polygons = []
        for label_int, label in enumerate(support_package["positive"].keys()):
            print(f"[{cnt}/{len(q_image_paths)}] Loading {qip} >>>")
            image = utils.import_image(qip)
            q_image_shape = image.shape
            image = cv2.resize(image, (cfg.window_size[0], cfg.window_size[1]))
            predictor.set_image(image)
            features = predictor.features

            similarity_maps = []
            for i, embedding in enumerate(support_package["positive"][label]):
                similarity_map = utils.get_similarity(embedding, features)
                # similarity_map = torch.where(similarity_map > cfg.threshold, 1., 0.)
                similarity_maps.append(similarity_map)
            for i, n_embedding in enumerate(support_package["negative"][label]):
                similarity_map = utils.get_similarity(n_embedding, features)
                # similarity_map = torch.where(similarity_map > cfg.threshold, 1., 0.)
                similarity_map = 1-similarity_map
                similarity_maps.append(similarity_map)

            similarity_maps = torch.stack(similarity_maps, dim=0)
            similarity_maps = torch.einsum('bij->ij', similarity_maps)/similarity_maps.shape[0]
            similarity_maps = torch.where(similarity_maps > cfg.threshold, 1., 0.)
            from skimage.morphology import erosion

            similarity_maps = erosion(similarity_maps.cpu().numpy(), np.ones((2, 2)))

            print(f"HIGHEST SIMILARITY FOUND: {np.max(similarity_maps)}")

            torchvision.utils.save_image(torch.from_numpy(similarity_maps), "./test.png")

            # local_max_coordinates = peak_local_max(similarity_maps, min_distance=7)
            local_max_coordinates = (similarity_maps > cfg.threshold).nonzero()
            print(f"LOCAL MAX COORDINATES in [Y, X] FORMAT: {local_max_coordinates}")

            local_max_coordinates = local_max_coordinates if local_max_coordinates.shape[0] > 0 else \
                (similarity_maps == torch.max(similarity_maps)).nonzero()

            if local_max_coordinates.shape[0] > 1:
                local_max_coordinates = [[elm, similarity_maps[elm[0], elm[1]].item()] for elm in local_max_coordinates]
                local_max_coordinates = sorted(local_max_coordinates, key=lambda x: x[1], reverse=True)
                med = median([elm[1] for elm in local_max_coordinates])
                local_max_coordinates = [elm[0] for elm in local_max_coordinates if elm[1] > cfg.threshold]

            print(f"LOCAL MAX COORDINATES in [Y, X] FORMAT: {local_max_coordinates}")

            for yx in local_max_coordinates:

                xy = utils.adapt_point(
                    {"x": yx[1], "y": yx[0]},
                    initial_shape=features.shape[-2:],
                    final_shape=image.shape[0:2]
                )

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
                    label=label,
                    polygon_resolution=cfg.labeling.polygon_resolution
                )

                masks.append(mask_)

                # create xml file from the coordinates
                coordinates = np.nonzero(mask_)
                y0, y1, x0, x1 = coordinates[0].min(), coordinates[0].max(), coordinates[1].min(), coordinates[1].max()
                bboxes.append([x0, y0, x1, y1])

                labels_str.append(label)
                labels_int.append(label_int)

        mask_path = qip.replace(
            cfg.data.query_dir,
            cfg.data.output_dir).replace(f".{cfg.data.format}", ".png")
        mask_dir = "\\".join(mask_path.split("\\")[:-1])
        if not os.path.isdir(mask_dir):
            os.makedirs(mask_dir)

        if len(masks) > 0:
            masks_transformed = utils.merge_multilabel_masks(masks, labels_int, COLORMAP=cfg.COLORMAP)

            cv2.imwrite(
                mask_path,
                masks_transformed * 255
            )

            annotations.create_xml_multilabel(image_name=qip, labels=labels_str, bboxes=bboxes)
            annotations.create_polygon_json(image_path=qip, polygons=polygons, size=masks_transformed.shape)
        else:
            print("NO MASK IS GENERATED FOR THIS IMAGE BASED ON THE GIVEN COORDINATES.")
            cv2.imwrite(
                mask_path,
                np.zeros((image.shape[0], image.shape[1], 3))
            )

        print("...")
