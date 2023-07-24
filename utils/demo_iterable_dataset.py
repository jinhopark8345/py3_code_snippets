
import torch
import math
from datasets import load_dataset
from itertools import islice
from torch.utils.data import DataLoader, get_worker_info


class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end

        # self.dataset = list(range(self.start, self.end))
        # breakpoint()


        # self.dataset = load_dataset(self.dataset, streaming=True)
        # self.dataset = load_dataset('json', data_files= 'my_file.json', streaming=True)
        dataset_name_or_path = "/home/jinho/Projects/tmp_repos/donut/dataset/MD-MED_donut_20230315_491_maxcollen17"
        self.dataset = load_dataset(dataset_name_or_path, split='train', streaming=True)
        # breakpoint()

    def prepare_input(self, sample):

        image, ground_truth = sample['image'], sample['ground_truth']

        # breakpoint()
        return ground_truth

    def __iter__(self):
        # breakpoint()

        worker_info = get_worker_info()
        print(worker_info)
        worker_total_num = worker_info.num_workers
        worker_id   = worker_info.id

        mapped_itr = map(self.prepare_input, self.dataset)
        mapped_itr = islice(mapped_itr, worker_id, None, worker_total_num)

        return  mapped_itr

            # per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            # worker_id = worker_info.id
            # iter_start = self.start + worker_id * per_worker
            # iter_end = min(iter_start + per_worker, self.end)
        # return iter(range(iter_start, iter_end))


# Define a `worker_init_fn` that configures each dataset copy differently
# def worker_init_fn(worker_id):
#     worker_info = torch.utils.data.get_worker_info()
#     dataset = worker_info.dataset  # the dataset copy in this worker process
#     overall_start = dataset.start
#     overall_end = dataset.end
#     # configure the dataset to only process the split workload
#     per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
#     worker_id = worker_info.id
#     dataset.start = overall_start + worker_id * per_worker
#     dataset.end = min(dataset.start + per_worker, overall_end)


if __name__ == '__main__':
    ds = MyIterableDataset(start=0, end=40)

    # print(list(ds))
    dl = DataLoader(
        dataset=ds, batch_size=2, num_workers=4
    )

    for e in dl:
        print(list(e))
