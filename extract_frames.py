import cv2 as cv
from pathlib import Path
import os
from tqdm import tqdm


def extract_images_vids(video_dir, img_dir, video_name, skip_frames):
    capture = cv.VideoCapture(video_dir)
    count = 0
    total = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            print("End of video reached")
            break
        if count % skip_frames == 0:
            img_name = "{0}_{1}.jpg".format(video_name, count)
            img_path = os.path.join(img_dir, img_name)
            cv.imwrite(img_path, frame)
            total += 1
            if total % 10 == 0:
                print("Images_written : ", total)
        count += 1
    print("Total : ", total)
    capture.release()


def main():
    videos_dir_2 = Path(r"C:\Users\fayaz\NTNU\Norsvin (13 October, 2022)")
    video_names = os.listdir(videos_dir_2)
    video_dirs = [os.path.join(videos_dir_2, name) for name in os.listdir(videos_dir_2)]
    image_dir = Path(r"C:\Users\fayaz\NTNU\barlow\train_images")
    for vid_name, videos_dir in tqdm(zip(video_names, video_dirs)):
        print("Reading : ", vid_name)
        extract_images_vids(videos_dir, image_dir, vid_name, 450)


if __name__ == "__main__":
    main()
