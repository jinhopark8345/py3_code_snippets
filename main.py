

from pdf2image_usage.pdf_utils import split_pdf_to_images
import os


def demo_split_pdf_to_images():
    root = os.path.realpath('.')
    src_path = os.path.join(root, "pdf2image_usage/data/src/lottery.pdf")
    dst_path = os.path.join(root, "pdf2image_usage/data/dst/")
    split_pdf_to_images(src_path, dst_path, dpi= 200)
