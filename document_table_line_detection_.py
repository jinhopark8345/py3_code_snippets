import numpy as np
import cv2
import os
from pprint import pprint


def find_edges(img_path):
    horizontal_line_threshold = 2
    hline_validator = lambda l: abs(l[0][1] - l[0][3]) < horizontal_line_threshold
    vertical_line_threshold = 10
    vline_validator = lambda l: abs(l[0][0] - l[0][2]) < vertical_line_threshold

    hline_ratio_to_doc_width = 0.5
    vline_ratio_to_doc_height = 0.1
    minHorizontalLineLength = None
    minVerticalLineLength = None
    maxLineGap = 3


    gray = cv2.imread(img_path)
    img_height, img_width, nchannels = gray.shape
    minHorizontalLineLength = (
        img_width * hline_ratio_to_doc_width if img_width is not None else 1000
    )

    minVerticalLineLength = (
        img_height * vline_ratio_to_doc_height if img_height is not None else 100
    )


    print(f'{minHorizontalLineLength, minVerticalLineLength = }')

    fval, sval = 50, 150
    edges = cv2.Canny(gray, fval, sval, apertureSize=3)

    file_path, ext = os.path.splitext(img_path)
    cv2.imwrite(f"{file_path}_edge_{fval}_{sval}_{1}{ext}", edges)

    hlines = cv2.HoughLinesP(
        image=edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        lines=np.array([]),
        minLineLength=minHorizontalLineLength,
        maxLineGap=maxLineGap,
    )

    vlines = cv2.HoughLinesP(
        image=edges,
        rho=1,
        theta=np.pi / 180,
        threshold=200,
        lines=np.array([]),
        minLineLength=minVerticalLineLength,
        maxLineGap=maxLineGap,
    )

    # check if line is alinged with document + check only the horizontal lines
    print(f"initial {len(hlines) = }")
    print(f"initial {len(vlines) = }")

    hlines_to_doc = filter(hline_validator, hlines)
    # vlines_to_doc = filter(vline_validator, vlines)

    for line in hlines_to_doc:
        x1, y1, x2, y2 = line[0]
        yval = (y1 + y2) / 2
        print(f"draw horizontal line, {line, yval = }")

        cv2.line(
            gray,
            (x1, y1),
            (x2, y2),
            (0, 0, 255), # red color
            3,
            cv2.LINE_AA,
        )

    # print(f'{len(list(vlines_to_doc)) = }')
    for line in vlines_to_doc:
        x1, y1, x2, y2 = line[0]
        print(f"draw vertical line, {line[0] = }")

        cv2.line(
            gray,
            (x1, y1),
            (x2, y2),
            (255, 0, 0), # blue
            3,
            cv2.LINE_AA,
        )


    cv2.imwrite(f"{file_path}_houghlines_{2}{ext}", gray)




def main():
    # pdf_path = "/home/jinho/Projects/asia-poc/data/attention.pdf"
    # split_pdf(pdf_path)

    # doc_path = "/home/jinho/Projects/asia-poc/data/out_8.jpg"
    # find_edges(doc_path)


if __name__ == "__main__":
    main()
