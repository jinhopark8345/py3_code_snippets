import os
import sh
from pdf2image import convert_from_path
from absl import app, flags, logging

flags.DEFINE_string("src", "", "src pdf path")
flags.DEFINE_string("dst", "", "output image dir")
flags.DEFINE_string("fmt", "png", "output image format")
flags.DEFINE_integer("dpi", 600, "dots per inch")

FLAGS = flags.FLAGS

"""
need poppler(pdf rendering library) and pdf2image package (python)
"""


def split_pdf_to_images(
    src_pdf_path: str,
    output_image_dir: str,
    output_image_format: str = "png",
    dpi: int = 600,
) -> None:

    """split pdf into images
    :param src_pdf_path: source pdf path
    :param output_image_dir: output image directory
    :param output_image_format: output image format
    :param dpi: output image dpi
    :returns: None

    """
    # make sure the dst folder exist
    os.makedirs(output_image_dir, exist_ok=True)

    # split pdf to pages,
    pages = convert_from_path(src_pdf_path, dpi=dpi, fmt=output_image_format)

    file_name, ext = os.path.splitext(src_pdf_path)
    for idx, page in enumerate(pages):
        output_image_path = f"{file_name}_{idx}.{output_image_format.lower()}"
        logging.info(f"save splited image on {output_image_path}")
        page.save(output_image_path, output_image_format)


def main(_):
    split_pdf_to_images(FLAGS.src, FLAGS.dst, FLAGS.fmt, FLAGS.dpi)


if __name__ == "__main__":
    app.run(main)
