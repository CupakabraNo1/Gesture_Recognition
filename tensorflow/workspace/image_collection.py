import os
import cv2

from constants import number_of_images, picture_labels, paths

for label in picture_labels:

    images_count = number_of_images
    images_label = label

    font = cv2.FONT_HERSHEY_PLAIN
    start = False

    label_name = os.path.join(paths['IMAGE_PATH'], images_label)
    count = image_name = 0

    try:
        os.mkdir(label_name)
    except FileNotFoundError or FileExistsError:
        image_name = len(os.listdir(label_name))
        images_count += len(os.listdir(label_name))

    video = cv2.VideoCapture(0)

    video.set(cv2.CAP_PROP_FRAME_WIDTH, 2000)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 2000)

    while True:
        ret, image = video.read()
        image = cv2.flip(image, 1)

        if not ret:
            continue

        if count == images_count:
            break

        if start:
            save_path = os.path.join(label_name, '{}_{}.jpg'.format(images_label, image_name + 1))
            cv2.imwrite(save_path, image)
            image_name += 1
            count += 1

        cv2.putText(image, "Label {}...Press Space [' '] key to start clicking pictures".format(images_label),
                    (20, 30), font, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, "Press 'q' to exit.",
                    (20, 60), font, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, "Image Count: {}".format(count),
                    (20, 100), font, 1, (12, 20, 200), 2, cv2.LINE_AA)
        cv2.imshow("Get Training Images", image)
        start = False
        k = cv2.waitKey(10)
        if k == ord('q'):
            break
        if k == ord(' '):
            start = not start

    video.release()
    cv2.destroyAllWindows()

# automation script to