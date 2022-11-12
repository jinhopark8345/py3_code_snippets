

def regex_split_demo(words):
    words = flatten_list(words)
    splited_word = []
    pattern = ",| |\/|\n|\.(?=[a-zA-Z])"
    for word in words:
        splited_word = re.split(pattern, word.strip())
        splited_word.extend(splited_word)
