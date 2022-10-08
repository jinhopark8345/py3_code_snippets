from PIL import ImageSequence, Image
from pprint import pprint




def method1(tiff_path):
    imgs = []
    with Image.open(tiff_path) as im:
        try:
            for _ in range(im.n_frames):
                im.seek(im.tell())
                imgs.append(im.copy())
        except EOFError:
            pass
    return imgs


def method2(tiff_path):
    imgs = []
    with Image.open(tiff_path) as im:
        try:
            while 1:
                im.seek(im.tell() + 1)
                imgs.append(im.copy())
        except EOFError:
            pass
    return imgs


def method3(tiff_path):
    imgs = []
    img = Image.open(tiff_path)
    for im in ImageSequence.Iterator(img):
        imgs.append(im.copy())

    return imgs


def main():
    multi_page_tiff_path = "/home/jinho/datasets/CDIP-compex-document-information-processing-dataset/imagesa/a/a/i/aai13a00/508950764+-0802.tif"

    pprint(method1(multi_page_tiff_path))
    print()
    pprint(method2(multi_page_tiff_path))
    print()
    pprint(method3(multi_page_tiff_path))
    print()

if __name__ == '__main__':
    main()
