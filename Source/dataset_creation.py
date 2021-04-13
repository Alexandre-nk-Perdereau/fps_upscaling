import cv2
import numpy
from PIL import Image
import os
from pathlib import Path
from os.path import join
from config import datafolder_path


def video_to_frames(video_path, path_output_dir, size=()):
    """
    create a folder containing all the images extracted from a video
    :param video_path:  (string) path of the video
    :param path_output_dir: (string)    path of the folder where the images will be stored
    :param size:    (int tupple) if void do nothing, else change the size of the images to this tupple. Useful to downsample image.
    """

    Path(path_output_dir).mkdir(parents=True, exist_ok=True)

    current_triplet = []
    vid_cap = cv2.VideoCapture(video_path)
    count = 0
    file_triplets = open(os.path.join(path_output_dir, 'triplets.txt'), 'w')

    while vid_cap.isOpened():
        success, image = vid_cap.read()

        if success:
            if size != ():
                image = Image.fromarray(image)
                image = image.resize(size)
                image = numpy.array(image)
            cv2.imwrite(os.path.join(path_output_dir, 'frame_%d.png') % count, image)
            if len(current_triplet) == 3:
                current_triplet[0] = current_triplet[1]
                current_triplet[1] = current_triplet[2]
                current_triplet[2] = "frame_" + str(count) + ".png"
                if count > 3:
                    string_triplets = ";" + current_triplet[0] + "," + current_triplet[1] + "," + current_triplet[2]
                else:
                    string_triplets = current_triplet[0] + "," + current_triplet[1] + "," + current_triplet[2]
                file_triplets.write(string_triplets)
            else:
                current_triplet.append("frame_" + str(count) + ".png")
            count += 1
            if count % 100 == 0:
                print(count)
        else:
            break
    cv2.destroyAllWindows()
    vid_cap.release()
    file_triplets.close()


def video_degradation(video_complete_path, output_name, divide_frame_factor, output_resolution):
    """
    create a video with degraded properties.
    :param video_complete_path: (string) path of the input video
    :param output_name: (string) name of the output video
    :param divide_frame_factor: (int) output_framerate = input_framerate / divide_frame_factor
    :param output_resolution:   (int tupple) resolution of the output video
    """
    savename = join(join(datafolder_path, "Videos"), output_name)
    vid_cap = cv2.VideoCapture(video_complete_path)
    input_fps = vid_cap.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(savename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), input_fps // 2, output_resolution)


    count = 0
    while vid_cap.isOpened():
        success, image = vid_cap.read()
        if success:
            count += 1
            if count >= divide_frame_factor:
                image = Image.fromarray(image)
                image = image.resize(output_resolution)
                image = numpy.array(image)
                writer.write(image)
                count = 0
        else:
            break
    vid_cap.release()
    writer.release()


if __name__ == '__main__':
    # video_to_frames(r"D:\github_local\videos\360p_sintel.mp4",
    #                 r"D:\github_local\fps_upscaling\Data\Datasets\360p_sintel")
    # video_to_frames(r"D:\github_local\videos\Spring - Blender Open Movie.mp4",
    #                 r"D:\github_local\frame_interpolation\Data\Datasets\240p_spring",
    #                 size=(426, 240))
    # video_to_frames(r"D:\github_local\videos\Sintel - Français - 3ème film libre de la fondat.mp4",
    #                 r"D:\github_local\frame_interpolation\Data\Datasets\240p_sintel",
    #                 size=(426, 240))
    # video_degradation(r"D:\github_local\videos\Spring - Blender Open Movie.mp4", "spring240pframedividedby2.avi", 2, (424, 240))
    video_to_frames(r"D:\github_local\videos\Spring - Blender Open Movie.mp4",
                    r"D:\github_local\frame_interpolation\Data\Datasets\480p_spring",
                    size=(720, 480))
    video_to_frames(r"D:\github_local\videos\Sintel - Français - 3ème film libre de la fondat.mp4",
                    r"D:\github_local\frame_interpolation\Data\Datasets\480p_sintel",
                    size=(720, 480))
