import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def template_matching_demo(
    input_image_path: str, template_image_path: str, output_image_path: str
):
    img_rgb = cv.imread(input_image_path)

    # resize original image to (1500, ?)
    img_rgb = cv.resize(
        img_rgb, dsize=(1500, int(img_rgb.shape[0] * 1500 / img_rgb.shape[1]))
    )

    # prepare for template matching : gray scale original image
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

    # read template as gray image
    template = cv.imread(template_image_path, 0)

    # get template shape
    # w, h = template.shape[::-1]

    result_img = cv.matchTemplate(
        img_gray, template, cv.TM_CCOEFF_NORMED
    )  # result_img : 0 ~ 1, float32, and shape isn't same as img_gray
    threshold = 0.90
    loc = np.where(result_img >= threshold)

    rect_width = 10

    """
        (Pdb) pp list(zip(*loc[::-1]))
        [(31, 43),
        (32, 43),
        (32, 44),
        (38, 46),
        (80, 120),
        (146, 120),
        (220, 120),
        (269, 120),
        ...
        ...
    """

    for pt in zip(*loc[::-1]):
        cv.rectangle(
            img_rgb,
            pt,
            (pt[0] + rect_width, pt[1] + rect_width),
            (0, 0, 255),
            2,
        )

    result_img = cv.normalize(
        result_img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U
    )

    vis = np.concatenate(
        (
            img_rgb,
            cv.resize(
                cv.cvtColor(result_img, cv.COLOR_GRAY2BGR),
                dsize = (img_rgb.shape[::-1][1:]), # img_rgb.shape : (width, height, n_channel)
            ),
            cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR),
        ),
        axis=1,
    )

    # breakpoint()
    cv.imwrite(output_image_path, vis)


def main():
    template_matching_demo(
        "images/input.jpg", "images/template.jpg", "images/output.jpg"
    )


if __name__ == "__main__":
    main()
