from pascal_voc_writer import Writer

from PIL import Image
# # create pascal voc writer (image_path, width, height)
# writer = Writer('path/to/img.jpg', 800, 598)

# # add objects (class, xmin, ymin, xmax, ymax)
# writer.addObject('truck', 1, 719, 630, 468)
# writer.addObject('person', 40, 90, 100, 150)

# # write to file
# writer.save('path/to/img.xml')

import xml.etree.ElementTree as ET

input_pascal_xml_path = "/home/jinho/datasets/CornerDetection/labels.pascalVOC/train2017/c.xml"
output_pascal_xml_path = "/home/jinho/datasets/CornerDetection/labels.pascalVOC/train2017_lt_br/c.xml"
import os

input_img_path = "/home/jinho/datasets/CornerDetection/images/train2017/c.png"
# parse xml file
tree = ET.parse(input_pascal_xml_path)
root = tree.getroot() # get root object

lt_boxes = []
rb_boxes = []
for member in root.findall('object'):
    class_name = member[0].text # class name

    # bbox coordinates
    xmin = int(member[4][0].text)
    ymin = int(member[4][1].text)
    xmax = int(member[4][2].text)
    ymax = int(member[4][3].text)

    lt_boxes.append((xmin-5, ymin-5, xmin+5, ymin+5))
    rb_boxes.append((xmax-5, ymax-5, xmax+5, ymax+5))

img = Image.open(input_img_path)
width, height = img.size

writer = Writer(input_img_path, width, height)

breakpoint()
for lt_box in lt_boxes:
    writer.addObject('left_top', lt_box[0], lt_box[1], lt_box[2], lt_box[3])
for rb_box in rb_boxes:
    writer.addObject('right_bottom', *rb_box)
writer.save(output_pascal_xml_path)
