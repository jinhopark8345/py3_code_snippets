
import cv2
import numpy as np
import glob

# frameSize = (500, 500)

img_array = []

img_dir = "/home/jinho/Downloads/meeting_with_Jaehyun/eng_video_images/2016_ella_exchange/"
for fn in list(glob.glob(f'{img_dir}*.JPG'))[:200]:
    img = cv2.imread(fn)
    h, w, c = img.shape
    size = (w, h)
    img_array.append(img)

out = cv2.VideoWriter('output_video.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 10, (1920, 1080))

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

# #     # print(filename)
# #     img = cv2.imread(filename)
# #     out.write(img)

# # out.release()

# import numpy as np
# import skvideo.io

# out_video =  np.empty([5, 1080, 1280, 3], dtype = np.uint8)
# out_video =  out_video.astype(np.uint8)

# # for f in
# for fn in glob.glob(f'{img_dir}*.JPG')[:10]:
# # for i in range(5):
#     img = cv2.imread(fn)
#     out_video[i] = img

# # Writes the the output image sequences in a video file
# skvideo.io.vwrite("video.mp4", out_video)
