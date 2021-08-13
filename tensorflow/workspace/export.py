import os
from constants import paths, files

# freezing the graph
FREEZE_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'exporter_main_v2.py ')
freeze = "python {} \
    --input_type=image_tensor \
    --pipeline_config_path={} \
    --trained_checkpoint_dir={} \
    --output_directory={}"\
    .format(
        FREEZE_SCRIPT,
        files['PIPELINE_CONFIG'],
        paths['CHECKPOINT_PATH'],
        paths['OUTPUT_PATH']
    )
print(freeze)
os.system(freeze)

# export tflite model
TFLITE_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'export_tflite_graph_tf2.py ')
tflite_graph= "python {} --pipeline_config_path={} --trained_checkpoint_dir={} --output_directory={}".format(TFLITE_SCRIPT ,files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'], paths['TFLITE_PATH'])
print(tflite_graph)
os.system(tflite_graph)

FROZEN_TFLITE_PATH = os.path.join(paths['TFLITE_PATH'], 'saved_model')
TFLITE_MODEL = os.path.join(paths['TFLITE_PATH'], 'saved_model', 'detect.tflite')

tflite = "tflite_convert \
--saved_model_dir={} \
--output_file={} \
--input_shapes=1,300,300,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=FLOAT \
--allow_custom_ops".format(FROZEN_TFLITE_PATH, TFLITE_MODEL, )
print(tflite)
os.system(tflite)

