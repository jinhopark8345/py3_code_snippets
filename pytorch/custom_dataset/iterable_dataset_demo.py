import torch
import math
from datasets import load_dataset, load_from_disk
import datasets
from itertools import islice
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import DonutProcessor
import json
import random
from typing import Any, List, Tuple

import torch
from torch.utils.data import Dataset

added_tokens = []

from transformers import DonutProcessor
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")

def get_iter_dataset(path):
    data = load_from_disk(path)
    def my_gen(n, data):
        for i in range(n):
            yield data[i]
    return datasets.IterableDataset.from_generator(my_gen, gen_kwargs={"n": len(data), "data": data})


class DonutDataset(IterableDataset):
    """
    PyTorch Dataset for Donut. This class takes a HuggingFace Dataset as input.

    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
    and it will be converted into pixel_values (vectorized image) and labels (input_ids of the tokenized string).

    Args:
        dataset_name_or_path: name of dataset (available at huggingface.co/datasets) or the path containing image files and metadata.jsonl
        max_length: the max number of tokens for the target sequences
        split: whether to load "train", "validation" or "test" split
        ignore_id: ignore_index for torch.nn.CrossEntropyLoss
        task_start_token: the special token to be fed to the decoder to conduct the target task
        prompt_end_token: the special token at the end of the sequences
        sort_json_key: whether or not to sort the JSON keys
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        max_length: int,
        split: str = "train",
        ignore_id: int = -100,
        task_start_token: str = "",
        prompt_end_token: str = None,
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token if prompt_end_token else task_start_token
        self.sort_json_key = sort_json_key

        # self.dataset = load_dataset(dataset_name_or_path, split=self.split)
        self.dataset = get_iter_dataset(dataset_name_or_path)

        seed, buffer_size = 42, 12
        self.dataset = self.dataset.shuffle(seed, buffer_size=buffer_size).with_format('torch')
        breakpoint()
        # self.dataset_length = len(self.dataset)

        # self.gt_token_sequences = []
        for sample in self.dataset:
            # ground_truth = json.loads(sample["ground_truth"])
            ground_truth = sample["ground_truth"]
            if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
                assert isinstance(ground_truth["gt_parses"], list)
                gt_jsons = ground_truth["gt_parses"]
            else:
                assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                gt_jsons = [ground_truth["gt_parse"]]

            tmp = [
                self.json2token(
                    gt_json,
                    update_special_tokens_for_json_key=self.split == "train",
                    sort_json_key=self.sort_json_key,
                )
                + processor.tokenizer.eos_token
                for gt_json in gt_jsons  # load json from list of json
            ]

            break

        self.add_tokens([self.task_start_token, self.prompt_end_token])
        self.prompt_end_token_id = processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)

    def json2token(self, obj: Any, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        self.add_tokens([fr"", fr""])
                    output += (
                        fr""
                        + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                        + fr""
                    )
                return output
        elif type(obj) == list:
            return r"".join(
                [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in added_tokens:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj

    def add_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings of the decoder
        """
        newly_added_num = processor.tokenizer.add_tokens(list_of_tokens)
        if newly_added_num > 0:
            model.decoder.resize_token_embeddings(len(processor.tokenizer))
            added_tokens.extend(list_of_tokens)
    # def __len__(self) -> int:
    #     return self.dataset_length

    def prepare_input(self, sample) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels
        Convert gt data into input_ids (tokenized string)
        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        # sample = self.dataset[idx]

        # inputs
        pixel_values = processor(sample["image"], random_padding=self.split == "train", return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze()

        # targets
        # target_sequence = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1

        ground_truth = sample["ground_truth"]
        if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
            assert isinstance(ground_truth["gt_parses"], list)
            gt_jsons = ground_truth["gt_parses"]
        else:
            assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
            gt_jsons = [ground_truth["gt_parse"]]

        target_sequence = [
            self.json2token(
                gt_json,
                update_special_tokens_for_json_key=self.split == "train",
                sort_json_key=self.sort_json_key,
            )
            + processor.tokenizer.eos_token
            for gt_json in gt_jsons  # load json from list of json
        ]

        target_sequence = target_sequence[0]

        input_ids = processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        labels = input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = self.ignore_id  # model doesn't need to predict pad token
        # labels[: torch.nonzero(labels == self.prompt_end_token_id).sum() + 1] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
        return pixel_values, labels, target_sequence

    def __iter__(self):
        worker_info = get_worker_info()
        # print(worker_info)
        if worker_info:
            worker_total_num = worker_info.num_workers
            worker_id   = worker_info.id
        else:
            worker_total_num = 1
            worker_id   = 0


        mapped_itr = map(self.prepare_input, self.dataset)
        mapped_itr = islice(mapped_itr, worker_id, None, worker_total_num)

        return  mapped_itr


class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, path):
        super(MyIterableDataset).__init__()
        self.dataset = get_iter_dataset(path)

    def prepare_input(self, sample):

        image, ground_truth = sample['image'], sample['ground_truth']
        breakpoint()

        # breakpoint()
        return ground_truth

    def __iter__(self):
        # breakpoint()

        worker_info = get_worker_info()
        print(worker_info)
        if worker_info:
            worker_total_num = worker_info.num_workers
            worker_id   = worker_info.id
        else:
            worker_total_num = 1
            worker_id   = 0


        mapped_itr = map(self.prepare_input, self.dataset)
        mapped_itr = islice(mapped_itr, worker_id, None, worker_total_num)

        return  mapped_itr

            # per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            # worker_id = worker_info.id
            # iter_start = self.start + worker_id * per_worker
            # iter_end = min(iter_start + per_worker, self.end)
        # return iter(range(iter_start, iter_end))


if __name__ == '__main__':
    # ds = MyIterableDataset(start=0, end=40)
    # ds = MyIterableDataset(start=0, end=40)

    # from pyinstrument import Profiler

    # profiler = Profiler()
    # profiler.start()
    ds = DonutDataset("/home/jinho/Downloads/save", 100)

    # print(list(ds))
    dl = DataLoader(
        dataset=ds, batch_size=1, num_workers=4
    )

    for _ in dl:
        ...

    # profiler.stop()
    # profiler.print()

    # for e in dl:
    #     print(e)
        # print(list(e))
