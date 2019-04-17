# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""label_image for tflite"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from PIL import Image

from tensorflow.lite.python import interpreter as interpreter_wrapper

def get_images(input_path):
  '''
  find image files in test data path
  return: list of files found
  '''
  LIST_EXT = ['csv', 'txt']
  IMAGES_EXT = ['jpg', 'png', 'jpeg', 'JPG']
  data_ext = os.path.splitext(input_path)[-1].replace('.', '')
  if data_ext in LIST_EXT:
    # csv format input
    with open(input_path, 'rt') as f:
      data = np.array(list(csv.reader(f)))
      files = data[:,0]
      gts = data[:,1]

  elif data_ext in IMAGES_EXT:
    # image format input
    files, gts = [input_path], None

  else:
    # folder format input
    files, gts = [], None
    for ext in IMAGES_EXT:
      files.extend([item for item in glob.glob(os.path.join(
        input_path, '*.{}'.format(ext))) if not os.path.splitext(item)[0].endswith('_act')])

  print('Find {} images'.format(len(files)))
  return files, gts

def load_labels(filename):
  my_labels = []
  input_file = open(filename, 'r')
  for l in input_file:
    my_labels.append(l.strip())
  return my_labels

if __name__ == "__main__":
  floating_model = False

  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--image", default="/tmp/grace_hopper.bmp", \
    help="image to be classified")
  parser.add_argument("-m", "--model_file", \
    default="/tmp/mobilenet_v1_1.0_224_quant.tflite", \
    help=".tflite model to be executed")
  parser.add_argument("-l", "--label_file", default="/tmp/labels.txt", \
    help="name of file containing labels")
  parser.add_argument("--input_mean", default=127.5, help="input_mean")
  parser.add_argument("--input_std", default=127.5, \
    help="input standard deviation")

  parser.add_argument("-o", "--output_csv", default="results.csv", \
    help="output csv path")

  args = parser.parse_args()

  interpreter = interpreter_wrapper.Interpreter(model_path=args.model_file)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # check the type of the input tensor
  if input_details[0]['dtype'] == np.float32:
    floating_model = True

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]

  images, gts = get_images(args.image)
  records, correct = [], 0
  for image_idx, image in tqdm(enumerate(images), total=len(images)):
    img = Image.open(image)
    img = img.resize((width, height))

    # add N dim
    input_data = np.expand_dims(img, axis=0)

    if floating_model:
      input_data = (np.float32(input_data) - args.input_mean) / args.input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(args.label_file)
    # for i in top_k:
    #   if floating_model:
    #     print('[{0}] {1:08.6f}: {2}'.format(image, float(results[i]), labels[i]))
    #   else:
    #     print('[{0}] {1:08.6f}: {2}'.format(image, float(results[i]/255.0), labels[i]))

    # if gts != None:
    #   records += list(zip([image] * len(results), results, labels, gts))
    # else:
    #   records += list(zip([image] * len(results), results, labels))

    max_idx = top_k[0]
    if gts is not None:
      gt = gts[image_idx]
      records.append([image, results[max_idx], labels[max_idx], gt, labels[max_idx] == gt])
      correct += int(labels[max_idx] == gt)
    else:
      records.append([image, results[max_idx], labels[max_idx]])

  pd.DataFrame.from_records(records).to_csv(args.output_csv, header=None, index=None)
  print("==> Saved outputs to {}".format(args.output_csv))

  if gts is not None:
    print("==> Accuracy: {}".format(float(correct) / len(images)))
