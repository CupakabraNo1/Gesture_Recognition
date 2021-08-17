import os
from constants import files, labels,paths, PRETRAINED_MODEL_NAME
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import shutil


# creating labelmap
with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

train = "{} -x {} -l {} -o {}" .format(
    files['TF_RECORD_SCRIPT'],
    os.path.join(paths['IMAGE_PATH'], 'train'),
    files['LABELMAP'],
    os.path.join(paths['ANNOTATION_PATH'], 'train.record')
)
print(train)
os.system(train)

test = "{} -x {} -l {} -o {}" .format(
    files['TF_RECORD_SCRIPT'],
    os.path.join(paths['IMAGE_PATH'], 'test'),
    files['LABELMAP'],
    os.path.join(paths['ANNOTATION_PATH'], 'test.record')
)
print(test)
os.system(test)

shutil.copy2(os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config'), os.path.join(paths['CHECKPOINT_PATH']))
print('Copied')

# modifying pipeline config
config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

pipeline_config.model.ssd.num_classes = len(labels)
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:
    f.write(config_text)

# training the model
TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
command = "python {} \
    --model_dir={} \
    --pipeline_config_path={} \
    --num_train_steps=2000".format(
    TRAINING_SCRIPT,
    paths['CHECKPOINT_PATH'],
    files['PIPELINE_CONFIG'])
print(command)
os.system(command)

# evaluate the model
evaluate = "python {} \
    --model_dir={} \
    --pipeline_config_path={} \
    --checkpoint_dir={}".format(
    TRAINING_SCRIPT,
    paths['CHECKPOINT_PATH'],
    files['PIPELINE_CONFIG'],
    paths['CHECKPOINT_PATH'])
print(evaluate)
os.system(evaluate)
os.system('exit()')
