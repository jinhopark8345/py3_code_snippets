import json
import os
from copy import deepcopy
import pandas as pd
from PIL import Image

import datasets
from datasets import load_dataset
from datasets import load_from_disk
from datasets import Dataset
from huggingface_hub import notebook_login, create_repo

HUGGING_FACE_TOKEN="<huggingface_token_you_can_find_from : https://huggingface.co/settings/tokens"
USER_ID = '<your huggingface user id>'

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

def train_split_to_train_and_validation_and_push_to_hub(train_path, test_path, save_path, repo_name):
    train = load_from_disk(train_path)
    test  = load_from_disk(test_path)

    # set size of test(validation) set
    # here we are spliting train set to train & validation set ( we already have test set )
    train_test_splited = train.train_test_split(test_size=100)
    train, val = train_test_splited['train'], train_test_splited['test']

    data = datasets.DatasetDict({
        'train': train,
        'val':val,
        'test': test
    })

    data.save_to_disk(save_path)
    data = load_from_disk(save_path)

    # make a repo and push the data to hub
    create_repo(repo_name, token=HUGGING_FACE_TOKEN, repo_type="dataset")
    data.push_to_hub(repo_id=f"{USER_ID}/{repo_name}", token=HUGGING_FACE_TOKEN)


if __name__ == "__main__":
    # SROIE : Scanned Receipts OCR and Information Extraction

    original_sroie_dataset_root = "/home/jinho/datasets/sroie"
    train_test_hug_dataset_save_root = "/home/jinho/datasets/sroie_huggingface"
    train_val_test_hug_dataset_save_root = "/home/jinho/datasets/sroie_huggingface_train_val_test"

    """
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

    # each json file has information of paired image
    # such as OCR info(bounding box, words) + labels (for tokenclassification)

    # we are going to
    # 1. parse them and save in hugginface dataset format (arrow format)
    # 2. split train dataset into train & validation dataset

    # after this we will be able to call/load dataset from huggingface hub with below command

    > from datasets import load_dataset
    > data = load_dataset(f"{USER_ID}/'sroie")

    """

    # 1. parse them and save in hugginface dataset format (arrow format)
    save_sroie_dataset_in_huggingface_arrow_format(original_sroie_dataset_root, train_test_hug_dataset_save_root)

    """

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

    # 2. split train dataset into train & validation dataset
    train_path = os.path.join(train_test_hug_dataset_save_root, 'train')
    test_path = os.path.join(train_test_hug_dataset_save_root, 'test')
    train_split_to_train_and_validation_and_push_to_hub(train_val_test_hug_dataset_save_root, train_path, test_path)


    """
    ❯ tree sroie_huggingface_train_val_test
    sroie_huggingface_train_val_test
    ├── dataset_dict.json
    ├── test
    │   ├── data-00000-of-00001.arrow
    │   ├── dataset_info.json
    │   └── state.json
    ├── train
    │   ├── data-00000-of-00001.arrow
    │   ├── dataset_info.json
    │   └── state.json
    └── val
        ├── data-00000-of-00001.arrow
        ├── dataset_info.json
        └── state.json

    3 directories, 10 files

    """

    # check if dataset is loaded correctly

    data = load_dataset(f"{USER_ID}/'sroie")

    """
    DatasetDict({
        train: Dataset({
            features: ['img', 'labels', 'words', 'bboxs'],
            num_rows: 526
        })
        test: Dataset({
            features: ['img', 'labels', 'words', 'bboxs'],
            num_rows: 347
        })
        val: Dataset({
            features: ['img', 'labels', 'words', 'bboxs'],
            num_rows: 100
        })
    })
    """
