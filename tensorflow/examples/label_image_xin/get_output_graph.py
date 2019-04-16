import json
import os
import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.python.framework import graph_util

from training.nets import nets_factory
from training.preprocessing import preprocessing_factory

TEN_CROP = True

data_dir = '/data5/xin/siemens/'
config = os.path.join(data_dir, 'siemens5_4k.json')
checkpoint_path = os.path.join(data_dir, 'model.ckpt-3360')
model_key = 'inception_resnet_v2'
labels_file = os.path.join(data_dir, 'siemens5_4k_irv2.json')
output_file = os.path.join(data_dir, 'test.pb' if not TEN_CROP else 'graph.pb')

def get_image_string_tensor_input_placeholder(eval_image_size, image_preprocessing_fn):
  """Returns input that accepts a batch of PNG or JPEG strings.
  Returns:
    a tuple of input placeholder and the output decoded images.
  """
  # batch_image_str_placeholder = tf.placeholder(
  #     dtype=tf.string,
  #     shape=[None],
  #     name='image_data')
  batch_image_str_placeholder = tf.placeholder(dtype=tf.float32, name='image_data')

  def decode_and_preprocess(encoded_image_string_tensor):
    # image_tensor = tf.image.decode_image(encoded_image_string_tensor,
    #                                      channels=3)
    image_tensor = encoded_image_string_tensor
    image_tensor.set_shape((None, None, 3))

    return image_preprocessing_fn(image_tensor, int(eval_image_size), int(eval_image_size))

  return (batch_image_str_placeholder,
          tf.map_fn(
              decode_and_preprocess,
              elems=batch_image_str_placeholder,
              dtype=tf.float32,
              parallel_iterations=32,
              back_prop=False))

labels = []
with open(labels_file) as f:
  jj = json.loads(f.read())
  labels = jj['labels'][0]
  # labels = jj['labels'] if FLAGS.multihead_fn else jj['labels'][0]
print('=> labels', labels)

with tf.Graph().as_default():
    # Get network_fn
    network_fn = nets_factory.get_network_fn(
      model_key,
      num_classes=len(labels))
      # L=width_param)

    # Build inference model.
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
      # name=FLAGS.preprocessing_name or FLAGS.model_key,
      name = model_key,
      is_training=False,
      ten_crop=TEN_CROP)

    # eval_image_size = FLAGS.eval_image_size or network_fn.get('default_image_size', 0) or 299
    # eval_image_size = network_fn.get('default_image_size', 0) or 299
    eval_image_size = 299

    # placeholder_tensor is the input strings
    # images_tensor is the decoded and preprocessed images
    placeholder_tensor, images_tensor = get_image_string_tensor_input_placeholder(eval_image_size, image_preprocessing_fn)
    if TEN_CROP:
      images_tensor = tf.reshape(images_tensor, [10, int(eval_image_size), int(eval_image_size), 3])
    print('==> images_tensor', images_tensor)


    # images_tensor = tf.placeholder(tf.float32, shape=[None, None, 3], name='image_data')
    # # image_contents = tf.read_file(image_data_placeholder)
    # # images_tensor = tf.image.decode_image(image_contents, channels=3)
    # # images_tensor.set_shape((None, None, 3))
    # images_tensor = image_preprocessing_fn(images_tensor, int(eval_image_size), int(eval_image_size))
    # images_tensor = tf.reshape(images_tensor, [1, int(eval_image_size), int(eval_image_size), 3])

    # images_tensor = tf.placeholder(
    #   dtype=tf.float32,
    #   name='image_data')
    # images_tensor.set_shape((None, None, 3))
    # images_tensor = image_preprocessing_fn(images_tensor, int(eval_image_size), int(eval_image_size))


    # Run inference.
    logits, _ = network_fn(images_tensor)
    print('==> logits', logits)
    if not TEN_CROP:
      out = tf.nn.softmax(logits, name='labels_softmax')
    else:
      out = tf.reduce_mean(tf.nn.softmax(logits), axis=0, name='labels_softmax')
    print('==> out', out)

      # sofxmax_predictions = []
      # for i in range(idx, idx+10):
      #   logits_lists = [logits[j][i].tolist() for j in range(len(class_sizes))]
      #   sofxmax_predictions.append(softmax(logits_lists)[0])
      # sofxmax_predictions = [sum(x) / float(len(x)) for x in zip(*sofxmax_predictions)]


    # dimensions_tensor = tf.constant([len(labels)])
    # class_tensor = tf.constant(labels)

    # Restore variables from training checkpoint.
    # checkpoint_path = '/models/{}.ckpt'.format(FLAGS.model_name)
    with tf.Session() as sess:
      try:
        var_list = slim.get_variables_to_restore()
        var_dict = {var.op.name: var for var in var_list}
        saver = tf.train.Saver(var_dict)
        saver.restore(sess, checkpoint_path)
        print('==> Successfully loaded model from %s' % checkpoint_path)
      except Exception as e:
        print('==> Error! %s' % e)
        raise Exception('No checkpoint file found at %s' % checkpoint_path)

      frozen_graph_def = graph_util.convert_variables_to_constants(
          sess, sess.graph.as_graph_def(), ['labels_softmax'])
      tf.train.write_graph(
          frozen_graph_def,
          os.path.dirname(output_file),
          os.path.basename(output_file),
          as_text=False)
      print('==> Saved frozen graph to %s' % output_file)

        # return

      # # Export inference model.
      # output_path = os.path.join(
      #     compat.as_bytes(FLAGS.output_dir),
      #     compat.as_bytes(FLAGS.model_name),
      #     compat.as_bytes(str(FLAGS.model_version)))
      # print 'Exporting trained model to', output_path
      # builder = saved_model_builder.SavedModelBuilder(output_path)

# sess = tf.InteractiveSession()
# saver = tf.train.Saver(tf.global_variables())
# saver.restore(sess, input_checkpoint_path)

# input_graph_def = sess.graph.as_graph_def()
# print('>>>>>>>>>> input_graph_def', input_graph_def)

# input_graph_path = '/data5/xin/siemens/graph.pbtxt'
# input_saver_def_path = ''
# input_binary = True
# input_checkpoint_path = '/data5/xin/siemens/model.ckpt-3360'

# output_node_names = "resnet_v2_152/Logits" 
# restore_op_name = "save/restore_all"
# filename_tensor_name = "save/Const:0"
# output_graph_path = os.path.join('/data5/xin/siemens/frozen_graph.pb')
# clear_devices = False

# initializer_nodes=''

# freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
#                           input_binary, input_checkpoint_path,
#                           output_node_names, restore_op_name,
#                           filename_tensor_name, output_graph_path,
#                           clear_devices, initializer_nodes)