
import os
root = "/home/jinho/datasets/CP-REF"
new_root = "/home/jinho/datasets/CP-REF_renamed/images"
fs = os.listdir(root)
import shutil

history = []
os.makedirs(new_root, exist_ok=True)
for idx, f in enumerate(fs):
    path = os.path.join(root, f)
    new_path = os.path.join(new_root, str(idx).zfill(5) + '.png')

    history.append(f'{path} -> {new_path}')
    shutil.copy(path, new_path)

breakpoint()
with open("history.txt", 'w') as f:
    f.writelines('\n'.join(history))
