import xml.etree.ElementTree as ET
import os
import xml.dom.minidom
import cv2
import numpy as np
import json
import base64


def create_xml_singlelabel(xmin, ymin, xmax, ymax, image_name, label):
    """
    Create Pascal VOC XML file from given input and save it with indentation.

    Args:
        xmin (int): Minimum x-coordinate of bounding box.
        ymin (int): Minimum y-coordinate of bounding box.
        xmax (int): Maximum x-coordinate of bounding box.
        ymax (int): Maximum y-coordinate of bounding box.
        image_name (str): Name of the image file.
        label (str): Label of the object in the bounding box.
    """
    # Create root element
    root = ET.Element('annotation')

    # Create filename element
    filename_elem = ET.SubElement(root, 'filename')
    filename_elem.text = image_name

    # Create object element
    object_elem = ET.SubElement(root, 'object')

    # Create name element
    name_elem = ET.SubElement(object_elem, 'name')
    name_elem.text = label

    # Create bounding box elements
    bndbox_elem = ET.SubElement(object_elem, 'bndbox')
    xmin_elem = ET.SubElement(bndbox_elem, 'xmin')
    xmin_elem.text = str(xmin)
    ymin_elem = ET.SubElement(bndbox_elem, 'ymin')
    ymin_elem.text = str(ymin)
    xmax_elem = ET.SubElement(bndbox_elem, 'xmax')
    xmax_elem.text = str(xmax)
    ymax_elem = ET.SubElement(bndbox_elem, 'ymax')
    ymax_elem.text = str(ymax)

    # Save XML to file with indentation
    xml_filename = os.path.splitext(image_name)[0] + '.xml'
    with open(xml_filename, 'wb') as xml_file:
        xml_str = ET.tostring(root)
        xml_dom = xml.dom.minidom.parseString(xml_str)
        # xml_file.write(xml_dom.toprettyxml(encoding='utf-8'))

    print(f'Created Pascal VOC XML file with indentation: {xml_filename}')
    return xml_dom.toprettyxml(encoding='utf-8')

def create_xml_multilabel(image_name, labels, bboxes):
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = 'images'
    ET.SubElement(annotation, 'filename').text = image_name
    ET.SubElement(annotation, 'path').text = os.path.abspath(image_name)

    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = '0'
    ET.SubElement(size, 'height').text = '0'
    ET.SubElement(size, 'depth').text = '0'

    ET.SubElement(annotation, 'segmented').text = '0'

    for label, bbox in zip(labels, bboxes):
        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = label
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'

        bbox_elem = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bbox_elem, 'xmin').text = str(bbox[0])
        ET.SubElement(bbox_elem, 'ymin').text = str(bbox[1])
        ET.SubElement(bbox_elem, 'xmax').text = str(bbox[2])
        ET.SubElement(bbox_elem, 'ymax').text = str(bbox[3])

    # Save XML to file with indentation
    xml_filename = os.path.splitext(image_name)[0] + '.xml'
    with open(xml_filename, 'wb') as xml_file:
        xml_str = (ET.tostring(annotation)).decode("utf-8")
        # xml_dom = xml.dom.minidom.parseString(xml_str)
        # xml_file.write(xml_dom.toprettyxml(encoding='utf-8'))

    print(f'Created Pascal VOC XML file with indentation: {xml_filename}')
    return xml_str

def generate_polygons_from_mask(polygons, mask, label, polygon_resolution):
    """
    Generate a list of polygons that encapsulate the ones in the binary mask.

    Args:
        mask (numpy.ndarray): The binary mask.

    Returns:
        list: A list of dictionaries, each containing the label and points of a polygon.
    """
    # Find the contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Generate polygons from the contours
    for i, contour in enumerate(contours):
        if int(len(contour)*polygon_resolution)>0:
            points = contour.squeeze()[np.arange(0,
                                                 len(contour),
                                                 int(len(contour)/int(len(contour)*polygon_resolution))
                                                 )].tolist()
            polygons.append({
                "label": label,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            })

    return polygons

def create_polygon_json(polygons, image_path, image_data, size=(1080,1440)):
    """
    Create a JSON file with the given list of polygons and image information.

    Args:
        polygons (list): A list of dictionaries, each containing the label and points of a polygon.
        image_path (str): The path to the image file.
        image_data (str, optional): The base64-encoded image data, if any. Defaults to ''.
        size (tuple, optional): The height and width of the image. Defaults to (1080,1440).
    """

    # Create the JSON data
    json_data = {
        "version": "5.1.1",
        "flags": {},
        "shapes": polygons,
        "imagePath": image_path,
        "imageData": image_data,
        "imageHeight": size[0],
        "imageWidth": size[1]
    }

    # Save the JSON data to a file
    # json_filename = os.path.splitext(image_path)[0] + '.json'
    # with open(json_filename, "w") as f:
    #     json.dump(json_data, f, indent=4)

    return json_data

