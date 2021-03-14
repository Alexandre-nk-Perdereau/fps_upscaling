from random import random

import cv2
import os


def video_to_frames(video_path, path_output_dir):
    vid_cap = cv2.VideoCapture(video_path)
    count = 0
    while vid_cap.isOpened():
        success, image = vid_cap.read()
        if success:
            cv2.imwrite(os.path.join(path_output_dir, 'frame%d.png') % count, image)
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    vid_cap.release()


if __name__ == '__main__':
    video_to_frames(r"D:\Star wars for Super Resolution\TheMandalorian_S2E1_1080pH264.mkv",
                    r"D:\github_local\fps_upscaling\Data\Datasets\TM_S2E1")
