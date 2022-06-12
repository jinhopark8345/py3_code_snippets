

from pdf2image_usage.pdf_utils import split_pdf_to_images
import os




# from config.definitions import ROOT_DIR
# print(os.path.join(ROOT_DIR, 'data', 'mydata.json'))

# root = os.path.relpath(os.path.join(os.getcwd(), "data"))
root = os.path.realpath('..')
print(root)
#
split_pdf_to_images("./pdf2image_usage/data/attenion.pdf", "./pdf2image_usage/data/output3", dpi= 200)
