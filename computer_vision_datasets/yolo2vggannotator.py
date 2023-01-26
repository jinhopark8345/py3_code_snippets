import json
import os

from PIL import Image


# Convert Yolo bb to Pascal_voc bb
def yolo_to_pascal_voc(x_center, y_center, w, h, image_w, image_h):
    w = w * image_w
    h = h * image_h
    x1 = ((2 * x_center * image_w) - w) / 2
    y1 = ((2 * y_center * image_h) - h) / 2
    x2 = x1 + w
    y2 = y1 + h
    return [x1, y1, x2, y2]


def yolo2json(
    image_root, annot_root, premade_json_path, output_json_path, point_json=True
):

    image_files = sorted(os.listdir(image_root))
    annot_files = sorted(os.listdir(annot_root))

    image2boxes = {}
    for image_file, annot_file in zip(image_files, annot_files):
        assert os.path.splitext(image_file)[0] == os.path.splitext(annot_file)[0]
        image_path = os.path.join(image_root, image_file)
        annot_path = os.path.join(annot_root, annot_file)
        width, height = Image.open(image_path).size

        voc_boxes = []
        with open(annot_path, "r") as f:
            lines = [l.strip().split() for l in f.readlines()]
            lines = [
                {"class": l[0], "yolo_box": tuple(map(float, l[1:]))} for l in lines
            ]
            for l in lines:
                pascal_voc_box = yolo_to_pascal_voc(*l["yolo_box"], width, height)
                pascal_voc_box = tuple(map(int, pascal_voc_box))
                voc_boxes.append(pascal_voc_box)
        image2boxes[image_file] = voc_boxes

    with open(premade_json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    filename2regions = {}
    for image_file, boxes in image2boxes.items():
        regions = []
        for box in boxes:
            cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
            region = {
                "shape_attributes": {"name": "point", "cx": cx, "cy": cy},
                "region_attributes": {},
            }
            regions.append(region)
        filename2regions[image_file] = regions

    for page_key, page in json_data.items():
        filename = page["filename"]
        regions = page["regions"]

        page["regions"] = filename2regions[filename]

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False)

    # with open("tmp.json", )


image_root = "/home/jinho/datasets/CornerDetection/CP-REF_renamed_101_478.autolabeling/images"
annot_root = "/home/jinho/datasets/CornerDetection/CP-REF_renamed_101_478.autolabeling/labels"
yolo2json(
    image_root,
    annot_root,
    premade_json_path="/home/jinho/Downloads/CP-REF_101_478_skeleton.json",
    output_json_path="/home/jinho/Downloads/CP-REF_101_478_skeleton_out.json",
)
