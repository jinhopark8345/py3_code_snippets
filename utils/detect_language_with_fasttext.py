import json
import fasttext
import re
from transformers import AutoTokenizer

lang_detector = fasttext.load_model('model/lid.176.ftz')




tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
pass_langs = set(['en', 'ko', 'ca', 'ja', 'zh'])


with open("/home/jinho/Projects/tmp_sciprts/tokenizer/tokenizer.json", 'r', encoding='utf8') as f:
    data = json.load(f)
vocab = data['model']['vocab']

count = 0
ko_pattern = re.compile(r'[ㄱ-ㅣ가-힣]')

# for item in vocab[-100:]:
for item in vocab:
    target_text = item[0]
    result = lang_detector.predict(target_text, k=2)


    first_lang = result[0][0][-2:]
    first_lang_confidence = result[1][0]

    korean_result = re.findall(ko_pattern, target_text)
    # if results:
    #     print(f'korean : {target_text}')
        # breakpoint()
    # pr
    if not korean_result and first_lang not in pass_langs \
        and first_lang_confidence > 0.8 and not any(c.isdigit() for c in target_text):
        count += 1
        print(target_text, result)

print(f'{count = }')

    # print(item)

# breakpoint()
