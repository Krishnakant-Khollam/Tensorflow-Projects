import os
cls = lambda: os.system("cls")

import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
config = config_util.get_configs_from_pipeline_file("models\pipeline.config")
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile("models\pipeline.config", "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)
labels = {'name':'mask', 'id':1}, {'name':'helmet', 'id':2}
pipeline_config.model.ssd.num_classes = len(labels)
pipeline_config.train_config.batch_size = 6
pipeline_config.train_config.fine_tune_checkpoint = "models\ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8\checkpoint\ckpt-0"
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= "models\label_map.pbtxt"
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = ['models\\train.record']
pipeline_config.eval_input_reader[0].label_map_path = "models\label_map.pbtxt"
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = ['models\\test.record']
config_text = text_format.MessageToString(pipeline_config)

with tf.io.gfile.GFile("models\pipeline.config", "wb") as f:
    f.write(config_text)
