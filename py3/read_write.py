from pprint import pprint
import fileinput


fpath = "./questinos.txt"
fpath2 = "./formatted_questions.txt"


with open(fpath, 'r') as f:
    lines = f.readlines()
    # lines = [l.strip() for l in lines]

new_lines = []
for l in lines:
    if len(l) > 0 and l[0].isdigit() and l.endswith("?\n"):
        new_lines.append("* "+ l)
    else:
        new_lines.append(l)

pprint(new_lines)
with open(fpath2, 'w') as f:
    f.writelines(new_lines)
