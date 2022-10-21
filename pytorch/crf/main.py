import torch
from torchcrf import CRF

num_tags = 5  # number of tags is 5
model = CRF(num_tags)

seq_length = 3  # maximum sequence length in a batch
batch_size = 2  # number of samples in the batch
emissions = torch.randn(seq_length, batch_size, num_tags)
tags = torch.tensor([[0, 1], [2, 4], [3, 1]], dtype=torch.long)  # (seq_length, batch_size)
print(model(emissions, tags))

emissions = torch.tensor([
    [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
    [[0, 0, 1, 0, 0], [0, 0, 0, 0, 1]],
    [[0, 0, 0, 1, 0], [0, 1, 0, 0, 0]],
])
tags = torch.tensor([[0, 1], [2, 4], [3, 1]], dtype=torch.long)  # (seq_length, batch_size)
print(model(emissions, tags))
breakpoint()

emissions = torch.tensor([
    [[100, 0, 0, 0, 0], [0, 100, 0, 0, 0]],
    [[0, 0, 100, 0, 0], [0, 0, 0, 0, 100]],
    [[0, 0, 0, 100, 0], [0, 100, 0, 0, 0]],
])
tags = torch.tensor([[0, 1], [2, 4], [3, 1]], dtype=torch.long)  # (seq_length, batch_size)
print(model(emissions, tags))

# tensor(-12.7431, grad_fn=<SumBackward0>)
emissions = torch.tensor([
    [[100, -100, -100, -100, -100]],
    [[-100, -100, 100, -100, -100]],
    [[-100, -100, -100, 100, -100]],
])
tags = torch.tensor([[0], [2], [3]], dtype=torch.long)
print(model(emissions, tags))

breakpoint()
