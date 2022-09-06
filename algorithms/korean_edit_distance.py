from collections import defaultdict
from pprint import pprint
from typing import List, Dict, Set, Tuple

from dictionary import hint_list, keyword2hint
from pred_input import pred_keys
import re


def jamo_levenshtein(s1, s2, debug=False):
    # https://lovit.github.io/nlp/2018/08/28/levenshtein_hangle/
    kor_begin = 44032
    kor_end = 55203
    chosung_base = 588
    jungsung_base = 28
    jaum_begin = 12593
    jaum_end = 12622
    moum_begin = 12623
    moum_end = 12643

    chosung_list = [
        "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ",
        "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
    ]

    jungsung_list = [
        "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ",
        "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
    ]

    jongsung_list = [
        " ", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ",
        "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ",
        "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
    ]

    jaum_list = [
        "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄸ", "ㄹ", "ㄺ", "ㄻ",
        "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅃ", "ㅄ", "ㅅ", "ㅆ", "ㅇ",
        "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
    ]

    moum_list = [
        "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ",
        "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
    ]

    def compose(chosung, jungsung, jongsung):
        char = chr(
            kor_begin
            + chosung_base * chosung_list.index(chosung)
            + jungsung_base * jungsung_list.index(jungsung)
            + jongsung_list.index(jongsung)
        )
        return char

    def decompose(c):
        i = ord(c)
        if jaum_begin <= i <= jaum_end:
            return (c, " ", " ")
        elif moum_begin <= i <= moum_end:
            return (" ", c, " ")

        # decomposition rule
        if character_is_korean(c):
            i -= kor_begin
            cho = i // chosung_base
            jung = (i - cho * chosung_base) // jungsung_base
            jong = i - cho * chosung_base - jung * jungsung_base
            return (chosung_list[cho], jungsung_list[jung], jongsung_list[jong])
        else:
            return (c, " ", " ")

    def character_is_korean(c):
        i = ord(c)
        return (
            (kor_begin <= i <= kor_end)
            or (jaum_begin <= i <= jaum_end)
            or (moum_begin <= i <= moum_end)
        )

    def levenshtein(s1, s2, cost=None, debug=False):
        if len(s1) < len(s2):
            return levenshtein(s2, s1, debug=debug)

        if len(s2) == 0:
            return len(s1)

        if cost is None:
            cost = {}

        # changed
        def substitution_cost(c1, c2):
            if c1 == c2:
                return 0
            return cost.get((c1, c2), 1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                # Changed
                substitutions = previous_row[j] + substitution_cost(c1, c2)
                current_row.append(min(insertions, deletions, substitutions))

            if debug:
                print(current_row[1:])

            previous_row = current_row

        return previous_row[-1]

    if len(s1) < len(s2):
        return jamo_levenshtein(s2, s1, debug)

    if len(s2) == 0:
        return len(s1)

    def substitution_cost(c1, c2):
        if c1 == c2:
            return 0
        return levenshtein(decompose(c1), decompose(c2)) / 3

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            # Changed
            substitutions = previous_row[j] + substitution_cost(c1, c2)
            current_row.append(min(insertions, deletions, substitutions))

        if debug:
            print(["%.3f" % v for v in current_row[1:]])

        previous_row = current_row

    return previous_row[-1]


def prediction_key2hint(
    prediction_key,
    hint_list: List[str],
    hint_set: Set[str],
    keyword2hint: Dict[str, str],
    min_dist_ratio:float = 0.5,
) -> Tuple[str, float]:
    # if input_key is already in hint keys
    if prediction_key in hint_set:
        return (prediction_key, 0)

    # replace with predefined keyword using distinctive keyword
    for keyword, predkey in keyword2hint.items():
        if keyword in prediction_key:
            return (keyword2hint[keyword], "shortcut")

    l = []
    for hint in hint_list:
        l.append((hint, jamo_levenshtein(prediction_key, hint, debug=False)))

    def get_two_smallest(l):
        m1 = m2 = float("inf")
        m1_str, m2_str = "", ""
        for ele in l:
            if ele[1] <= m1:
                m1, m2 = ele[1], m1
                m1_str, m2_str = ele[0], m1_str
            elif ele[1] < m2:
                m2 = ele[1]
                m2_str = ele[0]
        return (m1, m1_str), (m2, m2_str)

    (min_dist, min_dist_hint), (sec_min_dist, sec_min_dist_hint) = get_two_smallest(l)
    # print(f'{prediction_key, (min_dist, min_dist_hint), (sec_min_dist, sec_min_dist_hint)} = ')


    min_dist_threshold = min_dist_ratio * len(min(prediction_key, min_dist_hint, key=lambda s: len(s)))
    if min_dist != sec_min_dist and min_dist < min_dist_threshold:
        return (min_dist_hint, min_dist)
    else:
        return (prediction_key, "UNK")




def main():
    global pred_keys
    pred_keys = [s.replace(" ", "") for s in pred_keys]
    pred_keys = [re.sub(r'[~^0-9.\(\)\+\:\-·]', '', s) for s in pred_keys]
    hint2inputs = {hint: list() for hint in hint_list}
    hint2inputs["UNK"] = list()
    hint_set = set(hint_list)
    for pk in pred_keys:
        tmp = prediction_key2hint(pk, hint_list, hint_set, keyword2hint)

        if tmp[1] == "UNK":
            hint2inputs[tmp[1]].append(pk)
        elif tmp[1] == 'shortcut':
            hint2inputs[tmp[0]].append(pk)
        else:
            hint2inputs[tmp[0]].append(pk)

        print(f"{pk, tmp = }")
    pprint(hint2inputs)

if __name__ == '__main__':
    main()
