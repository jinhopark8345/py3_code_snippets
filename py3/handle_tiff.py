
import os
import cv2
from pprint import pprint
from tqdm import tqdm
import shutil
import exifread
from PIL import Image



def read_tiff_file(fpath):
    fname = os.path.basename(fpath)
    print(f'process: {fname}')
    rtv_str = ""

    try:
        pil_img = Image.open(fpath)
        rtv_str = pil_img.info
    except:
        print(f'failed to use Pillow on : {fpath}, using exifread')

        with open(fpath, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            rtv_str = tags

    return rtv_str


def main(root):
    fpaths = [os.path.join(root, f) for f in os.listdir(root)]
    for fpath in tqdm(fpaths):
        rtv = read_tiff_file(fpath)
        print(rtv)



if __name__ == '__main__':
    root = "/home/jinho/Projects/py3_code_snippets/py3/resources/tiff_files"
    main(root)
