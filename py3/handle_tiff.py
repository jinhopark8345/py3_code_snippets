
import os
import cv2
from pprint import pprint
from tqdm import tqdm
import exifread
from PIL import Image


root = "/home/jinho/Downloads/samples"
fpaths = [os.path.join(root, f) for f in os.listdir(root)]

def read_tiff_file(fpath):
    fname = os.path.basename(fpath)
    print(f'process: {fname}')
    try:
        pil_img = Image.open(fpath)
        print(f'\t {pil_img.info = }')
        # print()
        # breakpoint()

    # cv_img = cv2.imread(f, cv2.IMREAD_UNCHANGED)

    except:
        print(f'failed to use Pillow on : {fpath}, using exifread')

        with open(fpath, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            print(f'\t {tags = }')

            # pprint([(k, v.values) for k, v in tags.items()])

for fpath in tqdm(fpaths):
    read_tiff_file(fpath)
