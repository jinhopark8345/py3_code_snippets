from transformers import DistilBertTokenizer, DistilBertTokenizerFast

model_checkpoint = "distilbert-base-cased-distilled-squad"
slow_tokenizer = DistilBertTokenizer.from_pretrained(model_checkpoint)
fast_tokenizer = DistilBertTokenizerFast.from_pretrained(model_checkpoint)


sentence = "This sentence is not too long but we are going to split it anyway."

inputs = fast_tokenizer(
    sentence, truncation=True, return_overflowing_tokens=True, max_length=50
)
print(inputs["input_ids"])

inputs = fast_tokenizer(
    sentence, truncation=True, return_overflowing_tokens=True, max_length=6
)
print(inputs["input_ids"])
