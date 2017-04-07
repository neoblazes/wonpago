# Model for CNN estimator

import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

def model_fn(features, targets, mode, params):
  # Features are 13 of 9x9 and targets are [action, q] which represents Q(a).
  # Note that turn and ko info are set to valid_move map.
  # Original input was (-1, 13, 9, 9), will be reshaped to (-1, 9, 9, 13)
  # TODO: extract action from features.
  board = tf.transpose(tf.reshape(features, [-1, 13, 9, 9]), perm=[0, 2, 3, 1])
  # No relu, input includes negative. 13x25x128 = 41600
  conv_in = tf.layers.conv2d(inputs=board, filters=128, kernel_size=[5, 5],
      padding="same", activation=tf.nn.relu)
  conv1 = tf.layers.conv2d(inputs=conv_in, filters=64, kernel_size=[3, 3],
      padding="same", activation=tf.nn.relu) # 128*9*64 = 73728
  conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[3, 3],
      padding="same", activation=tf.nn.relu) # 64*9*64 = 36864
  conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[3, 3],
      padding="same", activation=tf.nn.relu) # 64*9*64 = 36864
  conv4 = tf.layers.conv2d(inputs=conv3, filters=64, kernel_size=[3, 3],
      padding="same", activation=tf.nn.relu) # 64*9*64 = 36864
  conv5 = tf.layers.conv2d(inputs=conv4, filters=64, kernel_size=[3, 3],
      padding="same", activation=tf.nn.relu) # 64*9*64 = 36864
  conv6 = tf.layers.conv2d(inputs=conv5, filters=64, kernel_size=[3, 3],
      padding="same", activation=tf.nn.relu) # 64*9*64 = 36864
  conv7 = tf.layers.conv2d(inputs=conv6, filters=64, kernel_size=[3, 3],
      padding="same", activation=tf.nn.relu) # 64*9*64 = 36864
  # TODO: start from swallow net.
  conv_out = tf.layers.conv2d(inputs=conv2, filters=2, kernel_size=[1, 1],
      padding="same") # To reduce size
  # Flattens conv2d output and use it as move probility.
  conv_flat = tf.reshape(conv_out, [-1, 9 * 9 * 2])

  # Dense layer and output.
  # 81*2*256 = 41472
  dense = tf.layers.dense(inputs=conv_flat, units=256)
  # Duel Q network, Q(a) = V + A(a) = Q2(a) or R (when terminal)
  value = tf.layers.dense(inputs=dense, units=1)
  # Actions, 256*81 = 20736
  a_net = tf.layers.dense(inputs=dense, units=82) # pass + 81 position
  q_net = a_net + value
  _, valid_move1, valid_move2, _ = tf.split(board, [7, 1, 1, 4], axis=3)
  valid = tf.reshape(valid_move1 + valid_move2, [-1, 9 * 9 * 1])

  predictions = {
    "value": value,
    "q_net": q_net,  # pass + 81
    "valid": valid,  # 81 positions, exclude pass
  }

  # For predict mode.
  if targets == None:
    return model_fn_lib.ModelFnOps(
      mode=mode,
      predictions=predictions)

  # For modes with targets.
  cols, target_qa = tf.split(tf.reshape(targets, [-1, 2]), [1, 1], axis=1)
  tf.range(tf.shape(board)[0])
  predict_qa = tf.gather_nd(q_net,
      tf.stack((tf.range(tf.shape(board)[0]),
                         tf.to_int32(tf.reshape(cols, [-1]))), -1))
  loss = tf.losses.mean_squared_error(target_qa, predict_qa)

  eval_metric_ops = {
    "accuracy"  : tf.metrics.accuracy(targets, logits, name='accuracy'),
    "precision" : tf.metrics.precision(targets, logits, name='precision')
  }

  train_op = tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=params["learning_rate"],
      optimizer="SGD")

  return model_fn_lib.ModelFnOps(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops)

def GetEstimator(model_dir, config=None, params={}):
  if not "learning_rate" in params.keys():
    params["learning_rate"] = 0.0001
  return tf.contrib.learn.Estimator(
    model_dir=model_dir, config=config, model_fn=model_fn, params=params)
