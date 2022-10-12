import cv2 as cv
import os


def create_video(image_folder, video_name):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    print(images)
    frame = cv.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv.VideoWriter(video_name, 0, 20, (width, height))

    for image in images:
        video.write(cv.imread(os.path.join(image_folder, image)))

    cv.destroyAllWindows()
    video.release()
