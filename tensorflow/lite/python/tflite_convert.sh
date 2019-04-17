MODEL=irv2
ROOT=/data5/xin/tflite/${MODEL}
CKPT=/data5/xin/ocr/train/receipts_good_bad_30k_irv2/receipts_good_bad_30k_irv2.ckpt


# to inference graph
cd ~/Hive/hive-image-classification/training
cp /data2/eason/tf_records/receipts_good_bad_30k_horovod/labels.json ./
python export_inference_graph.py --alsologtostderr --model_name=inception_resnet_v2 --image_size=299 --output_file=${ROOT}/${MODEL}.pb


# verify output node name
bazel run tensorflow/python/tools:inspect_checkpoint --  --file_name ${CKPT} | grep softmax


# to frozen model
bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph=${ROOT}/${MODEL}.pb \
  --input_checkpoint=${CKPT} \
  --input_binary=true \
  --output_graph=${ROOT}/frozen_${MODEL}.pb \
  --output_node_names=softmax


# to tflite
tflite_convert \
--output_file=${ROOT}/frozen_${MODEL}.tflite \
--graph_def_file=${ROOT}/frozen_${MODEL}.pb \
--input_arrays=input \
--output_arrays=softmax


# test it
python /workspace/xli/tensorflow/tensorflow/lite/examples/python/label_image.py -m ${ROOT}/frozen_${MODEL}.tflite -l ${ROOT}/labels.txt