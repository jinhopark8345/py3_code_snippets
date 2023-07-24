
from transformers import AutoTokenizer, XLMRobertaModel
from transformers import XLMRobertaTokenizer
from transformers import AutoTokenizer
import torch

def train():
    old_tokenizer = AutoTokenizer.from_pretrained("/home/jinho/Projects/tmp_repos/donut/result/train_md_med/20230402_233942")

    new_tokens = ['괄', '펠', '멸', '닐', '퇴', '좌', '톨흡', '졸', '멜', '촬', '캡', '톱', '텍', '엠', '받', '딘', '퓨', '흡', '겔', '엽', '럽', '1', '템', '넥', '낮', '템1', '펜', '롤', '헐', '롬', '팩', '흉', '섭', '및']


    with open("/home/jinho/Projects/tmp_sciprts/texts/faker_and_mdmed_corpus_20230414-mecab.txt", 'r') as f:
        training_corpus = f.readlines()
    tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 60000)

    # pretrained_model.
    tokenizer.save_pretrained("test-tokenizer")




    # tokenizer.save_vocabulary("./tokenizer_save")
    breakpoint()

def test():

    # print(len(pretrained_model.decoder.tokenizer))
    # pretrained_model.decoder.tokenizer.add_tokens(new_tokens)
    # print(len(pretrained_model.decoder.tokenizer))

    # new_vocab_len = len(pretrained_model.decoder.tokenizer)
    # pretrained_model.resize_token_embeddings(new_vocab_len)
    tokenizer_test = AutoTokenizer.from_pretrained("/home/jinho/Projects/tmp_sciprts/test-tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    new_tokens = ['괄', '펠', '멸', '닐', '퇴', '좌', '톨흡', '졸', '멜', '촬', '캡', '톱', '텍', '엠', '받', '딘', '퓨', '흡', '겔', '엽', '럽', '1', '템', '넥', '낮', '템1', '펜', '롤', '헐', '롬', '팩', '흉', '섭', '및']


    test = "'이게무야ㅕㅑ및쳣어?'"

    for nt in new_tokens:
        print(nt, tokenizer.tokenize(nt), tokenizer.encode(nt))


    breakpoint()


test()
