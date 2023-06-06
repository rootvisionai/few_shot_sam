from PIL import Image
import numpy as np
import base64
import io

def import_image(path):
    image_ = Image.open(path)
    image = Image.new("RGB", image_.size)
    image.paste(image_)
    image = np.asarray(image)
    return image

def adapt_point(pts, initial_shape, final_shape):
    scale_y = final_shape[0] / initial_shape[0]
    scale_x = final_shape[1] / initial_shape[1]
    pts_ = {}
    pts_["y"] = pts["y"] * scale_y
    pts_["x"] = pts["x"] * scale_x
    return pts_

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

def get_image(image_data):
    # base64 encoded string
    image_data = base64.b64decode(image_data)
    image_ = Image.open(io.BytesIO(image_data))
    image = Image.new("RGB", image_.size)
    image.paste(image_)
    return image
