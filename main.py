import glob

import numpy as np
import torch
from PIL import Image
import os
import cv2

from segment_anything import sam_model_registry, SamPredictor
import interface
import pascal_voc
import utils


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

    xy_list = []
    labels_str = []
    while True:
        # Support Images: Load, point, extract
        s_image_paths = glob.glob(os.path.join(cfg.data.support_dir, f"*.{cfg.data.format}"))
        embeddings = []
        for qip in s_image_paths:
            image = np.asarray(Image.open(qip))
            image = cv2.resize(image, (cfg.window_size[0], cfg.window_size[1]))
            points = interface.click_on_point(img=image)
            if not None in points:
                for pt in points:
                    embedding = utils.get_embedding(predictor, image, pt)
                    embeddings.append(embedding)

        # Query Images: Load, extract, match, segment
        q_image_paths = glob.glob(os.path.join(cfg.data.query_dir, f"*.{cfg.data.format}"))
        for cnt, qip in enumerate(q_image_paths):
            print(f"[{cnt}/{len(q_image_paths)}] Loading {qip} >>>")
            q_image = np.asarray(Image.open(qip))
            q_image_shape = q_image.shape
            q_image = cv2.resize(q_image, (cfg.window_size[0], cfg.window_size[1]))
            predictor.set_image(q_image)
            features = predictor.features

            similarity_maps = []
            for i, embedding in enumerate(embeddings):
                similarity_map = utils.get_similarity(embedding, features)
                similarity_map = torch.where(similarity_map > cfg.threshold, 1., 0.)
                similarity_maps.append(similarity_map)

            similarity_maps = torch.stack(similarity_maps, dim=0)
            similarity_maps = torch.einsum('bij->ij', similarity_maps)
            print(f"HIGHEST SIMILARITY: {torch.max(similarity_maps).item()}")

            yx = (similarity_maps == torch.max(similarity_maps)).nonzero()
            xy = utils.adapt_point(
                {"x": yx[0, 1].item(), "y": yx[0, 0].item()},
                initial_shape=features.shape[-2:],
                final_shape=q_image.shape[0:2]
            )

            xy_list.append(np.array([[xy["x"], xy["y"]]]).astype(int))

        label, event = interface.relabel_or_continue()
        labels_str.append(label)
        if event == "END":
            break
        elif event == "CANCEL":
            labels_str = labels_str[0:-1]
            break
        else:
            pass

    masks = []
    bboxes = []
    for i in range(len(labels_str)):
        l_ = np.ones((1,))

        mask_, scores, logits = predictor.predict(
            point_coords=xy_list[i],
            point_labels=l_,
            multimask_output=True,
        )
        mask_ = mask_.astype(np.uint8)
        mask_ = cv2.resize(mask_[0], (q_image_shape[1], q_image_shape[0]))
        masks.append(mask_)

        # create xml file from the coordinates
        coordinates = np.nonzero(mask_)
        y0, y1, x0, x1 = coordinates[0].min(), coordinates[0].max(), coordinates[1].min(), coordinates[1].max()
        bboxes.append([x0, y0, x1, y1])

    pascal_voc.create_file_multilabel(image_name=qip, labels=labels_str, bboxes=bboxes)

    masks_transformed = utils.merge_multilabel_masks(masks, COLORMAP=cfg.COLORMAP)

    cv2.imwrite(
        qip.replace(cfg.data.query_dir,
                    cfg.data.output_dir).replace(f".{cfg.data.format}",
                                                 ".png"),
        masks_transformed * 255
    )