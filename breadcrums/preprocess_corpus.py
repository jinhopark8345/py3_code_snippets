import re
import json



def preprocess_kowiki():
    path = "/home/jinho/Projects/tmp_repos/donut/synthdog/resources/corpus/kowiki_l.txt"

    texts =[]
    with open(path, 'r') as f:
        texts.extend(f.readlines())

    exs = texts[0].split()


    # Regular expression to match words in parentheses
    # pattern = re.compile(r'\((\w+)\)')
    closing_pattern = re.compile(r'[(\[{]([^\])}]+)[)\]}]')
    split_chars = r"[-/ %()\[:]"
    word_spliting_pattern = re.compile(r'[-/ %()\[:(\d+)]')
    # split_chars = [split_chars[i] for i in range(len(split_chars))]


    with open("output.txt", 'w', encoding='utf8') as file:
        for text in exs:
            cur_words = []

            ori_text = text

            # preprocess text
            text = text.replace("\`", "").replace("\?", "")

            # split with closing pattern
            remaining_words = closing_pattern.sub(" ", text)
            cur_words.append(remaining_words) # add remaining words
            for match in closing_pattern.findall(text):
                words = match.split()
                cur_words.extend(words)

            new_cur_words = []
            for word in cur_words:
                words = word_spliting_pattern.split(word)
                new_cur_words.extend(words)

            new_cur_words2 = []
            for word in new_cur_words:
                new_cur_words2.extend(word.split())

            new_cur_words =  new_cur_words2

            new_cur_words = [w.strip().strip("_") for w in new_cur_words]
            new_cur_words = [w for w in new_cur_words if w != ""]

            # check
            # for word in new_cur_words:
            #     for schar in split_chars:
            #         if schar in word:
            #             print(word)
            #             assert len(word) == 1

            print(f'{ori_text} -> {new_cur_words}', file=file)

        """
        1. add special characters: '{', "}", '(', ")", "[", "]", ":",
        2. currency: $, \, ...
        3. add measures: cm, mm, ym, ...

        """

