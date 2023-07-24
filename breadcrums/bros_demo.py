import torch
from bros import BrosTokenizer, BrosModel


tokenizer = BrosTokenizer.from_pretrained("naver-clova-ocr/bros-base-uncased")
model = BrosModel.from_pretrained("naver-clova-ocr/bros-base-uncased")


width, height = 1280, 720

words = ["to", "the", "moon!"]
quads = [
    [638, 451, 863, 451, 863, 569, 638, 569],
    [877, 453, 1190, 455, 1190, 568, 876, 567],
    [632, 566, 1107, 566, 1107, 691, 632, 691],
]

bbox = []
for word, quad in zip(words, quads):
    n_word_tokens = len(tokenizer.tokenize(word))
    bbox.extend([quad] * n_word_tokens)

cls_quad = [0.0] * 8
sep_quad = [width, height] * 4
bbox = [cls_quad] + bbox + [sep_quad]

encoding = tokenizer(" ".join(words), return_tensors="pt")
input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]

bbox = torch.tensor([bbox])
bbox[:, :, [0, 2, 4, 6]] = bbox[:, :, [0, 2, 4, 6]] / width
bbox[:, :, [1, 3, 5, 7]] = bbox[:, :, [1, 3, 5, 7]] / height

outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask)
last_hidden_state = outputs.last_hidden_state

print("- last_hidden_state")
print(last_hidden_state)
print()
print("- last_hidden_state.shape")
print(last_hidden_state.shape)
