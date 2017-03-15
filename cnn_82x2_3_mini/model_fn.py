# Model for CNN estimator

import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

def model_fn(features, targets, mode, params):
  # Features areboard[81] + liberty[81] + last_move[1] + ko[2].
  # Extracts 9x9x2 from board and liberty map for conv2d.
  board_features, last_move, ko = tf.split(features, [81*2, 1, 2], axis=1)  
  board = tf.reshape(board_features, [-1, 9, 9, 2])
  conv1 = tf.layers.conv2d(inputs=board, filters=64, kernel_size=[5, 5],
      padding="same", activation=tf.nn.relu) # out 9x9
  conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[3, 3],
      padding="same", activation=tf.nn.relu)
  conv3 = tf.layers.conv2d(inputs=conv2, filters=32, kernel_size=[3, 3],
      padding="same", activation=tf.nn.relu)
  conv4 = tf.layers.conv2d(inputs=conv3, filters=1, kernel_size=[1, 1],
      padding="same", activation=tf.nn.relu) # maybe for border?
  # Flattens conv2d output and attaches last_move info.
  conv_flat = tf.concat([tf.reshape(conv4, [-1, 9 * 9 * 1]), last_move], 1)

  # Dense layer and output.
  dense1 = tf.layers.dense(inputs=conv_flat, units=64, activation=tf.nn.relu)
  dense2 = tf.layers.dense(inputs=dense1, units=64, activation=tf.nn.relu)
  output_layer = tf.contrib.layers.linear(dense2, 1)
  predictions = tf.reshape(output_layer, [-1])

  # For predict mode.
  if targets == None:
    return model_fn_lib.ModelFnOps(
      mode=mode,
      predictions=predictions)

  # For modes with targets.
  loss = tf.losses.mean_squared_error(targets, predictions)

  eval_metric_ops = {
    "rmse": tf.metrics.root_mean_squared_error(
        tf.cast(targets, tf.float32), predictions)
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

def GetEstimator(model_dir, config=None):
  config = tf.contrib.learn.RunConfig()
  config.tf_config.gpu_options.allow_growth=True
  model_params = {"learning_rate": 0.001}
  return tf.contrib.learn.Estimator(
    model_dir=model_dir, config=config, model_fn=model_fn, params=model_params)
