from pdf2image_demo.split_pdf import split_pdf_to_images
import os


def demo_split_pdf_to_images():
    root = os.path.realpath(".")
    src_path = os.path.join(root, "pdf2image_demo/data/src/lottery.pdf")
    dst_path = os.path.join(root, "pdf2image_demo/data/dst/")
    split_pdf_to_images(src_path, dst_path, dpi=200)


def main():
    demo_split_pdf_to_images()


if __name__ == "__main__":
    main()
