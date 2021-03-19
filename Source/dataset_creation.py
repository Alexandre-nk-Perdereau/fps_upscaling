import cv2
import os


def video_to_frames(video_path, path_output_dir):

    current_triplet = []
    vid_cap = cv2.VideoCapture(video_path)
    count = 0
    file_triplets = open(os.path.join(path_output_dir, 'triplets.txt'), 'w')

    while vid_cap.isOpened():
        success, image = vid_cap.read()
        if success:
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



if __name__ == '__main__':
    video_to_frames(r"D:\blender_videos\test.mkv",
                    r"D:\github_local\fps_upscaling\Data\Datasets\test_ds")
