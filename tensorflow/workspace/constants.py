import os

# images to collect
picture_labels = ["serbia","like","palm"]

# number of images per each category
number_of_images = 20

# names
CUSTOM_MODEL_NAME = 'my_ssd_mobnet_v2_fpnlite'
# CUSTOM_MODEL_NAME = 'my_centernet'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
# PRETRAINED_MODEL_NAME = 'centernet_mobilenetv2_fpn_od'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'


paths = {
    'WORKSPACE_PATH': os.path.join('tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('tensorflow', 'scripts'),
    'APIMODEL_PATH': os.path.join('tensorflow', 'models'),
    'ANNOTATION_PATH': os.path.join('tensorflow', 'workspace', 'annotations'),
    'IMAGE_PATH': os.path.join('tensorflow', 'workspace', 'images'),
    'MODEL_PATH': os.path.join('tensorflow', 'workspace', 'models'),
    'PRETRAINED_MODEL_PATH': os.path.join('tensorflow', 'workspace', 'pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME),
    'OUTPUT_PATH': os.path.join('tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'export'),
    'TFJS_PATH': os.path.join('tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'tfjsexport'),
    'TFLITE_PATH': os.path.join('tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'tfliteexport'),
    'PROTOC_PATH': os.path.join('tensorflow', 'protoc')
}

files = {
    'PIPELINE_CONFIG': os.path.join('tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# labels
labels = [
    {'name': 'palm', 'id': 1},
    {'name': 'like', 'id': 2},
    {'name': 'serbia', 'id': 3}
]

for path in paths.values():
    if not os.path.exists(path):
        os.makedirs(path)


