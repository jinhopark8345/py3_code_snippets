import json
import os
from PIL import Image


def lrbr_box2yolo_box(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]

img_root = "/home/jinho/datasets/CornerDetection/tmp/images"
label_root = "/home/jinho/datasets/CornerDetection/tmp/labels"
os.makedirs(label_root, exist_ok=True)

with open("/home/jinho/datasets/CornerDetection/tmp/CP-REF_renamed_100_out_checked_till100.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

for page_key, page in data.items():
    fname = page['filename']
    regions = page['regions']

    img_path = os.path.join(img_root, fname)
    img = Image.open(img_path)
    width, height = img.size

    fbasename = os.path.splitext(fname)[0]
    label_path = os.path.join(label_root, fbasename + '.txt')

    result = []
    for region in regions:

        cx, cy = region['shape_attributes']['cx'], region['shape_attributes']['cy']

        box_height = min(int(height * 0.015),40)
        box_half_height = box_height // 2
        bbox = [cx-30, cy-box_half_height, cx+30, cy+box_half_height]


        yolo_bbox = lrbr_box2yolo_box(bbox, width, height)
        bbox_string = " ".join([str(x) for x in yolo_bbox])
        result.append(f"0 {bbox_string}")

    if result:
        # generate a YOLO format text file for each xml file
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(result))

