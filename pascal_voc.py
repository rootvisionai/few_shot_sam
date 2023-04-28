import xml.etree.ElementTree as ET
import os

import xml.dom.minidom

def create_file_singlelabel(xmin, ymin, xmax, ymax, image_name, label):
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
        xml_file.write(xml_dom.toprettyxml(encoding='utf-8'))

    print(f'Saved Pascal VOC XML file with indentation: {xml_filename}')

def create_file_multilabel(image_name, labels, bboxes):
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
        xml_str = ET.tostring(annotation)
        xml_dom = xml.dom.minidom.parseString(xml_str)
        xml_file.write(xml_dom.toprettyxml(encoding='utf-8'))

    print(f'Saved Pascal VOC XML file with indentation: {xml_filename}')