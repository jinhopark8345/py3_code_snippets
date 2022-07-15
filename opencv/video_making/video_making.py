import cv2
import numpy as np
import glob
import os
from collections import OrderedDict
from typing import Any
import logging

size = (1920, 1080)
fps = 10

max_dur = 1.0
min_dur = 0.3

def save_obj_as_pickle(save_path: str, data: Any):
    import pickle
    with open(save_path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(load_path: str):
    import pickle
    with open(load_path, "rb") as f:
        loaded_data = pickle.load(f)
    return loaded_data

def save_img_as_pickle():
    pick_dir = "/home/jinho/Downloads/picks"
    img_array = []
    max_img = 1000
    cnt = 0

    dir_list = [
        "2016_ella_exchange_student",
        "2017_me_norway_exchange_st",
        "2018_ella_korea_easter",
        "2018_me_norway_before_airforce",
        "2020_me_norway_after_airforce",
        "2021_2022_ella_korea",
    ]


    imgname_img_dict = OrderedDict()


    for dir_name in dir_list:
        dir_path = os.path.join(pick_dir, dir_name)

        for f_name in sorted((os.listdir(dir_path))):
            fn = os.path.join(dir_path, f_name)
            img = cv2.imread(fn)
            ori_h, ori_w, _ = img.shape
            new_size = (int(ori_w / (ori_h / 1080)), 1080)
            left_width = int((size[0] - new_size[0]) / 2)
            img = cv2.resize(img, new_size)
            left_padding = np.zeros((size[1], left_width, 3), np.uint8)
            right_padding = np.zeros((size[1], left_width, 3), np.uint8)
            logging.info(f"reading {fn = } ...")
            img = np.concatenate((left_padding, img, right_padding), axis=1)
            imgname_img_dict[f_name] = img
            cnt += 1

            if cnt > max_img:
                break
        if cnt > max_img:
            break

    save_obj_as_pickle("imgarr.pickle", imgname_img_dict)

def make_video(pickle_path, output_path ="./project.mp4" ):
    img_dict = load_pickle(pickle_path)
    # breakpoint()

    first_img = "IMG_0914.JPG"
    last_img = "20220408_151413.jpg"

    keys = list(img_dict.keys())

    keys.remove(first_img)
    keys.remove(last_img)

    sequence_list = []
    first_half_keys = [first_img] + keys

    idx2nimgs = get_idx2nimgs(min_dur, max_dur, first_half_keys)

    first_half = []
    for key, nimgs in zip(first_half_keys, idx2nimgs):
        first_half += [key] * nimgs


    sequence_list.extend([first_img]* int(5 * fps))
    # sequence_list.extend(keys)
    sequence_list.extend(first_half)
    sequence_list.extend([last_img] * int(20 * fps))
    # sequence_list.extend(reversed(first_half))



    # sequence_list.extend(list(reversed(keys)))
    # sequence_list.append(first_img)

    # self._fourcc = VideoWriter_fourcc(*'MP4V')
    # cv2.VideoWriter_fourcc('a','v','c','1')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MP4V"), fps, size)
    for key in sequence_list:
        out.write(img_dict[key])
    out.release()

def get_idx2nimgs(min_dur, max_dur, img_arr):
    # l = list(range(1, 11))
    n_img_arr = len(img_arr)

    assert n_img_arr > 1

    duration_gap = (max_dur - min_dur ) / (n_img_arr - 1)

    cur_dur = max_dur
    idx2nimgs = []
    for i in range(n_img_arr):
        nimgs = int(cur_dur * fps)
        # print(f'{cur_dur, nimgs = }')
        idx2nimgs.append(nimgs)

        cur_dur -= duration_gap
    return idx2nimgs

def main():

    # save_img_as_pickle()
    make_video("./imgarr.pickle")



if __name__ == '__main__':
    main()




