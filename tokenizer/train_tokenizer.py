import argparse
import collections
import os
import random
import re
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# import sentencepiece
import sentencepiece as spm
import torch
from datasets import load_dataset
from mecab import MeCab
from transformers import T5Tokenizer
from bs4 import BeautifulSoup



def something1():
    corpus_paths = [
        "/home/jinho/Projects/tmp_repos/donut/synthdog/resources/corpus/kowiki_l.txt",
        "/home/jinho/Projects/tmp_repos/donut/synthdog/resources/corpus/kowiki.txt",
        "/home/jinho/Projects/tmp_repos/donut/synthdog/resources/corpus/enwiki.txt",
    ]

    all_lines = []

    for path in corpus_paths:
        with open(path, "r") as f:
            all_lines.extend(f.readlines())
    breakpoint()


def make_corpus_with_mecab(input_path, output_path):
    # morph 단위로 분할된 말뭉치 생성
    mecab = MeCab()
    with open(output_path, "w") as o_f:
        with open(input_path) as f:
            lines = f.readlines()
            for line in lines:
                lien = line.strip()
                tokens = mecab.morphs(line)
                string = " ".join(tokens)
                o_f.write(string)
                o_f.write("\n")


def train_sentencepiece(corpus, prefix, required_chars, vocab_size=52000):
    """
    sentencepiece를 이용해 vocab 학습
    :param corpus: 학습할 말뭉치
    :param prefix: 저장할 vocab 이름
    :param vocab_size: vocab 개수
    """
    spm.SentencePieceTrainer.train(
        input=corpus,
        vocab_size=vocab_size,
        model_prefix=prefix,
        # input_senetnece_size=100000000,
        shuffle_input_sentence=True,
        split_digits=True,
        model_type="bpe",
        character_coverage=1.0,
        # required_chars=required_chars,
        # f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" +  # 7은 특수문자 개수
        # " --model_type=unigram" +
        # " --max_sentence_length=999999" +  # 문장 최대 길이
        # " --pad_id=0 --pad_piece=[PAD]" +  # pad token 및 id 지정
        # " --unk_id=1 --unk_piece=[UNK]" +  # unknown token 및 id 지정
        # " --bos_id=2 --bos_piece=[BOS]" +  # begin of sequence token 및 id 지정
        # " --eos_id=3 --eos_piece=[EOS]" +  # end of sequence token 및 id 지정
        # " --user_defined_symbols=[SEP],[CLS],[MASK]" +  # 기타 추가 토큰 SEP: 4, CLS: 5, MASK: 6
        # " --input_sentence_size=100000" +  # 말뭉치에서 셈플링해서 학습
        # " --shuffle_input_sentence=true" + # 셈플링한 말뭉치 shuffle
        # " --split_digits=True"
    )


required_chars = [
    "~",
    "`",
    "!",
    "@",
    "#",
    "$",
    "%",
    "^",
    "&",
    "*",
    "(",
    ")",
    "-",
    "_",
    "+",
    "=",
    "{",
    "}",
    "[",
    "]",
    "|",
    ":",
    ";",
    "\\",
    "'",
    '"',
    "\/",
    "<",
    ">",
    ",",
    ".",
    "?",
    "⓪",
    "①",
    "②",
    "③",
    "④",
    "⑤",
    "⑥",
    "⑦",
    "⑧",
    "⑨",
    "⑩",
    "⑪",
    "⑫",
    "⑬",
    "⑭",
    "⑮",
    "⑯",
    "⑰",
    "⑱",
    "⑲",
    "⑳",
    "㉑",
    "㉒",
    "㉓",
    "㉔",
    "㉕",
    "㉖",
    "㉗",
    "㉘",
    "㉙",
    "㉚",
    "㉛",
    "㉜",
    "㉝",
    "㉞",
    "㉟",
    "㊱",
    "㊲",
    "㊳",
    "㊴",
    "㊵",
    "㊶",
    "㊷",
    "㊸",
    "㊹",
    "㊺",
    "㊻",
    "㊼",
    "㊽",
    "㊾",
    "㊿",
    "⓿",
    "❶",
    "❷",
    "❸",
    "❹",
    "❺",
    # "❻", "❼", "❽", "❾", "❿", "⓫", "⓬", "⓭", "⓮", "⓯", "⓰", "⓱",
    # "⓲", "⓳", "⓴", "☆", "★", "⭐️", "🌟", "💫", "✨", "✳️", "✴️",
    # "♡", "♥", "♧", "♣", "♤", "♠", "ㆍ", "∙", "•", "◦", "※", "✓",
    # "✔", "☑", "☒", "→", "←", "↑", "↓", "↔", "↕", "↗", "↙", "↖", "↘", "⇄", "⇆", "⇒",
    # "⇏", "⇐", "⇑", "⇓", "⇔", "➜", "➡", "➤", "⇦", "⇧", "⇨", "⇩", "🔚", "🔙", "🔛",
    # "🔝",
    # "🔜",
    # "☚", "☛", "☜", "☝", "☞", "☟", "◇", "◆", "□", "■", "◈", "▣", "△", "▲", "▽", "▼",
    # "◁", "◀", "▷", "▶", "○", "●", "⊙", "◐", "◑", "◎", "￦", "$", "💲", "￡", "￥",
    # "€", "￠", "₠", "₡", "₢", "₣", "₤", "₥", "₦", "₧", "₨", "₪", "₫",
    # "₭", "₮", "₰", "₱", "¼", "½", "¾", "⅓", "⅔", "⅕", "⅖", "⅗", "⅘", "⅙", "⅚", "⅛", "⅜",
    # "⅝", "⅞", "⊕", "⊖", "⊗", "≠", "≡", "≤", "≥", "≪≫", "≺≻", "⊂⊃", "∀", "∁", "∂", "∅",
    # "∆", "∇", "∈", "∉", "∋", "∌", "∍", "∎", "∏", "∐", "∑", "−", "∧∨", "∩",
    # "∪", "√", "∛", "∜", "∝", "∞", "∫", "㎧", "㎨", "㎅", "㎆", "㎇", "㎈", "㎉", "㏒",
    # "㎍", "㎎", "㎏", "㎕", "㎖", "㎗", "㎘", "㎛", "㎜", "㎝", "㎞", "㎟", "㎠", "㎡", "㎢", "㎣",
    # "㎤", "㎥", "㎦", "㎲", "㎳", "㎐", "㎑", "㎒", "㎓", "㏈", "㏊", "㏗", "㏘", "㎭", "㏀", "㏁",
]

# breakpoint()


# make_corpus_with_mecab()
corpus = [
    "/home/jinho/Projects/tmp_sciprts/texts/faker_and_mdmed_corpus_20230414-mecab.txt",
    "/home/jinho/Projects/tmp_sciprts/texts/kowiki.txt",
    "/home/jinho/Projects/tmp_sciprts/texts/enwiki.txt",
]

# train_sentencepiece(corpus, "kowiki_jinho/4", list(set(required_chars)), 52000)

# spm_vocab = spm.SentencePieceProcessor()
# spm_vocab.load(os.path.join(/home/jinho/Projects/tmp_sciprts/kowiki-mecab.txt, "kowiki_32000.model"))


def train_tokenizer():

    ###########################
    # data_dir = './data'
    # dataset = load_dataset('nsmc')
    # os.makedirs(data_dir, exist_ok=True)
    # for split_key in dataset.keys():
    #     doc_path = f"{data_dir}/{split_key}.txt"
    #     with open(doc_path, 'w') as f:
    #         for doc in dataset[split_key]['document']:
    #             f.write(doc+'\n')
    ###########################

    # ###########################
    # data_dir = './data'
    # paths = [str(x) for x in Path(data_dir).glob("*.txt")]
    # corpus = ",".join(paths)
    # prefix = "kowiki_jinho/t5-sp-bpe-nsmc4"
    # vocab_size = 31900-7
    # required_chars = ".,|"
    # spm.SentencePieceTrainer.train(
    #     f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" +
    #     f"--required_chars={required_chars}" +
    #     " --model_type=bpe" +
    #     " --max_sentence_length=999999" + # 문장 최대 길이 (너무 길면 에러발생)
    #     " --pad_id=0 --pad_piece=<pad>" + # pad (0)
    #     " --unk_id=1 --unk_piece=<unk>" + # unknown (1)
    #     " --bos_id=2 --bos_piece=<s>" + # begin of sequence (2)
    #     " --eos_id=3 --eos_piece=</s>" + # end of sequence (3)
    #     " --byte_fallback=true" + # add byte_fallback for unk tokens
    #     " --split_digits=true" +
    #     " --user_defined_symbols=<sep>,<cls>,<mask>") # 사용자 정의 토큰
    # ###########################

    # ###########################
    tokenizer = T5Tokenizer(vocab_file="kowiki_jinho/t5-sp-bpe-nsmc4.model")
    tokenizer.save_pretrained("kowiki_jinho/t5-tokenizer-bpe-nsmc4")
    lines = [
        "`DEVOCEAN`은 SK그룹의 대표 개발자 커뮤니티이자🧑",
        "내/외부 개발자 간 소통과 성장을 위한 플랫폼을 상징합니다.👋",
        "`Developers`' Ocean 개발자들을 위한 영감의 바다🙏",
        "`Devotion` 헌신,몰두,전념💯",
        "`Technology for Everyone` 모두를 위한 기술👍",
    ]
    for line in lines:
        tokens = tokenizer.tokenize(line)
        inputs = tokenizer(line)
        decoded_sequence = tokenizer.decode(inputs['input_ids'])
        print(line) # 입력 데이터
        # print(tokens)  # subword로 토큰화된 데이터
        print(inputs['input_ids'])
        print(list(zip(tokens, inputs['input_ids'])))
        print(decoded_sequence) # subword토큰화된 데이터 -> token id -> 복원된데이터
        print()
    # ###########################

    # tokenizer = T5Tokenizer(vocab_file="kowiki_jinho/t5-sp-bpe-nsmc2.model")
    # tokenizer.save_pretrained("kowiki_jinho/t5-tokenizer-bpe-nsmc2")

    # with open("/home/jinho/Projects/tmp_sciprts/texts/ko_mdmed.txt", 'r') as f:
    #     data = f.read()

    # breakpoint()



def parse_sth():

    all_lines = []


    # paths = [str(x) for x in Path(data_dir).glob("*.txt")]
    paths = [str(x) for x in Path("/home/jinho/Downloads/kowiki-20230401-pages-articles/kowiki-20230401-pages-articles_wikiextracted").rglob("*wiki_*", )]


    # word_spliting_pattern = re.compile(r'[-/ %()\[:(\d+)]')
    # pattern1 = r"[<includeonly>,<Includeonly>].*?[</includeonly>,</Includeonly>]"
    CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    ban_words= ["includeonly", "text-align", "stxt-ageft"]
    ban_words_exact_match = ["|-", ""]

    # for path in paths:
    for path in paths[:2]:
        with open(path, 'r') as f:
            data = f.read()
        soup = BeautifulSoup(data, 'html.parser')
        # doc_content = soup.find('doc').text.strip()
        texts = [e.text.strip() for e in soup.findAll('doc')]

        new_texts = []
        for t in texts:
            new_texts.extend(t.splitlines())
        # texts = [t.splitlines() for t in texts]
        texts = [t for t in new_texts if t != ""]

        new_texts = []
        for t in texts:
            # if re.match(pattern1, t):
                # print(t)
                # breakpoint()

            # remaining_words = closing_pattern.sub(" ", text)
            t = CLEANR.sub("", t)
            new_texts.append(t)


        texts = new_texts
        texts = [t for t in texts if all([bool(ban_word not in t) for ban_word in ban_words]) ]
        texts = [t for t in texts if all([bool(ban_word != t) for ban_word in ban_words_exact_match]) ]
        # breakpoint()
        all_lines.extend(texts)

    # breakpoint()
    with open("/home/jinho/Downloads/something.txt", 'w') as f:
        # f.writelines(all_lines)
        f.writelines(l + '\n' for l in all_lines)

    # breakpoint()

def main():
    ...
    # parse_sth()
    # train_tokenizer()
    input_path = "/home/jinho/Downloads/something.txt"
    output_path = "/home/jinho/Downloads/something2.txt"
    make_corpus_with_mecab(input_path, output_path)

if __name__ == '__main__':
    main()
