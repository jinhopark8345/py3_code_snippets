import json
from pprint import pprint

# loading
with open('resources/json_demo.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
pprint(data)

# dumping
with open("resources/json_save_output.json", 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)
