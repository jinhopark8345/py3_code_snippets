import json
import os
from copy import deepcopy

import pandas as pd
from datasets import Dataset, load_from_disk
from PIL import Image


def save_sroie_dataset_in_huggingface_arrow_format(sroie_root, save_root):
    # DATE, ADDRESS, COMPANY, TOTAL
    # --------------------------------
    # B-DATE, I-DATE
    # B-ADDRESS, I-ADDRESS
    # B-COMPANY, I-COMPANY
    # B-TOTAL, I-TOTAL
    # O

    class_names = {"DATE", "ADDRESS", "COMPANY", "TOTAL"}
    splits = ["train", "test"]

    for split in splits:
        save_split_path = os.path.join(save_root, split)
        split_path = os.path.join(sroie_root, split)
        annos_dir = os.path.join(split_path, "tagged")
        imgs_dir = os.path.join(split_path, "images")
        annos = sorted(os.listdir(annos_dir))
        imgs = sorted(os.listdir(imgs_dir))
        dicts = []

        for anno, img in zip(annos, imgs):
            anno_file = os.path.basename(anno)
            img_file = os.path.basename(img)
            anno_path = os.path.join(annos_dir, anno)
            img_path = os.path.join(imgs_dir, img)

            assert os.path.splitext(anno_file)[0] == os.path.splitext(img_file)[0]

            # prepare data
            img = Image.open(img_path)
            with open(anno_path, "r") as f:
                data = json.load(f)
            labels = data["labels"]

            # remove B, I tags from original labels
            new_labels = deepcopy(labels)
            for lidx, l in enumerate(labels):
                for c in class_names:
                    if l.endswith(c):
                        new_labels[lidx] = c
                        break

            d = {
                "imgs": img,
                "labels": new_labels,
                "words": data["words"],
                "bbox": data["bbox"],
            }
            dicts.append(d)

        dataset_dicts = {
            "img": [d["imgs"] for d in dicts],
            "labels": [d["labels"] for d in dicts],
            "words": [d["words"] for d in dicts],
            "bboxs": [d["bbox"] for d in dicts],
        }

        dataset = Dataset.from_dict(dataset_dicts)
        dataset.save_to_disk(save_split_path)


if __name__ == "__main__":
    sroie_root = "/home/jinho/datasets/sroie"
    save_root = "/home/jinho/datasets/sroie_huggingface"

    """
    # SROIE : Scanned Receipts OCR and Information Extraction

    # if you download sroie dataset, it should look like below

    sroie
    ├── test
    │   ├── images
    │   │   ├── X00016469670.jpg
    │   │   ├── X00016469671.jpg
    │   │   ├── X51005200931.jpg
    │   │   ├── X51005230605.jpg
            ...
    │   └── tagged
    │       ├── X00016469670.json
    │       ├── X00016469671.json
    │       ├── X51005200931.json
    │       ├── X51005230605.json
            ...
    └── train
        ├── images
        │   ├── X00016469612.jpg
        │   ├── X00016469619.jpg
        │   ├── X00016469620.jpg
        │   ├── X00016469622.jpg
        ...
        └── tagged
            ├── X00016469612.json
            ├── X00016469619.json
            ├── X00016469620.json
            ├── X00016469622.json
            ...


    # after running below method you will have

    ❯ tree sroie_huggingface
    sroie_huggingface
    ├── test
    │   ├── data-00000-of-00001.arrow
    │   ├── dataset_info.json
    │   └── state.json
    └── train
        ├── data-00000-of-00001.arrow
        ├── dataset_info.json
        └── state.json

    2 directories, 6 files


    """

    save_sroie_dataset_in_huggingface_arrow_format(sroie_root, save_root)
