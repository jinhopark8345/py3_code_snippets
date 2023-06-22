import datasets
from datasets import load_dataset
from datasets import load_from_disk
from huggingface_hub import notebook_login, create_repo



HUGGING_FACE_TOKEN=""

# merge two datases

train_path = "/Users/jinho/Projects/bros/datasets/hug_datasets/train"
val_path = "/Users/jinho/Projects/bros/datasets/hug_datasets/validation"
train = load_from_disk(train_path)
val  = load_from_disk(val_path)
data = datasets.DatasetDict({'train': train, 'val':val})

# (Pdb++) datasets.DatasetDict({'train': train, 'val':val})
# DatasetDict({
#     train: Dataset({
#         features: ['parse', 'meta', 'words'],
#         num_rows: 526
#     })
#     val: Dataset({
#         features: ['parse', 'meta', 'words'],
#         num_rows: 100
#     })
# })

# make a repo and push the data to hub
create_repo("bros-sroie", token=HUGGING_FACE_TOKEN, repo_type="dataset")
data = load_from_disk("/Users/jinho/Projects/bros/datasets/hug_datasets/together")
data.push_to_hub(repo_id="jinho8345/bros-sroie", token=HUGGING_FACE_TOKEN)
