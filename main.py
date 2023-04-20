import glob

import torchvision.utils
import time
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
        torchvision.utils.save_image(similarity_maps, f"{time.time()}.png")
        print(f"HIGHEST SIMILARITY: {torch.max(similarity_maps).item()}")

        yx = (similarity_maps == torch.max(similarity_maps)).nonzero()
        xy = utils.adapt_point(
            {"x": yx[0, 1].item(), "y": yx[0, 0].item()},
            initial_shape=features.shape[-2:],
            final_shape=q_image.shape[0:2]
        )

        xy = np.array([[xy["x"], xy["y"]]]).astype(int)
        labels = np.ones((1,))

        masks, scores, logits = predictor.predict(
            point_coords=xy,
            point_labels=labels,
            multimask_output=True,
        )
        masks = masks.astype(np.uint8)[0]
        masks = cv2.resize(masks,
                           (q_image_shape[1], q_image_shape[0]))

        # create xml file from the coordinates
        coordinates = np.nonzero(masks)
        y0, y1, x0, x1 = coordinates[0].min(), coordinates[0].max(), coordinates[1].min(), coordinates[1].max()
        pascal_voc.create_file(xmin=x0, ymin=y0, xmax=x1, ymax=y1, image_name=qip, label=cfg.labeling.label)

        cv2.imwrite(
            qip.replace(cfg.data.query_dir,
                        cfg.data.output_dir).replace(f".{cfg.data.format}",
                                                     ".png"),
            masks*255
        )