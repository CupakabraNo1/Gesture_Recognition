import cv2
import numpy as np
import tensorflow as tf
import time
from load_model import detect_fn

from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
from constants import files

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)
num_of_frames = 120
frame_start = time.time()

while cap.isOpened():

    ret, frame = cap.read()
    image = np.array(frame)
    input_tensors = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
    inference_time_start = time.time()
    detections = detect_fn(input_tensors)
    inference_time_stop = time.time()

    sec = inference_time_stop - inference_time_start
    print("Inference time: {}".format(sec))
    num_of_frames -= 1
    if num_of_frames == 0:
        frame_end = time.time()
        fps = 120 / ( frame_end - frame_start )
        num_of_frames = 120
        frame_start = time.time()

    num_detections = int(detections.pop('num_detections'))
    detections = {
        key:value[0, :num_detections].numpy() for key, value in detections.items()
    }

    detections['num_detections'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image.copy()

    print("Frames per second: {0}".format(fps))

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'].astype(np.int64) + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=2,
        min_score_thresh=.8,
        agnostic_mode=False
    )

    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
