import xml.etree.ElementTree as ET
import os

import xml.dom.minidom

def create_file(xmin, ymin, xmax, ymax, image_name, label):
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

    # Create XML tree
    tree = ET.ElementTree(root)

    # Save XML to file with indentation
    xml_filename = os.path.splitext(image_name)[0] + '.xml'
    with open(xml_filename, 'wb') as xml_file:
        xml_str = ET.tostring(root)
        xml_dom = xml.dom.minidom.parseString(xml_str)
        xml_file.write(xml_dom.toprettyxml(encoding='utf-8'))

    print(f'Saved Pascal VOC XML file with indentation: {xml_filename}')
