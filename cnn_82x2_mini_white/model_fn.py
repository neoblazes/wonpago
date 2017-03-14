# Model for DNN regressor

import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

def model_fn(features, targets, mode, params):
  # Build 9x9x2 from board and liberty map
  print(features)
  first_hidden_layer = tf.contrib.layers.relu(features, 10)
  board = tf.reshape(features, [-1, 9, 9, 2])
  conv1 = tf.layers.conv2d(inputs=board, filters=64, kernel_size=[5, 5],
      padding="same", activation=tf.nn.relu) # out 9x9
  conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[3, 3],
      padding="same", activation=tf.nn.relu)
  conv3 = tf.layers.conv2d(inputs=conv2, filters=32, kernel_size=[3, 3],
      padding="same", activation=tf.nn.relu)
  conv4 = tf.layers.conv2d(inputs=conv3, filters=1, kernel_size=[1, 1],
      padding="same", activation=tf.nn.relu) # maybe for border?
  conv_flat = tf.reshape(conv4, [-1, 9 * 9 * 1])
  dense = tf.layers.dense(inputs=conv_flat, units=256, activation=tf.nn.relu)
  output_layer = tf.contrib.layers.linear(dense, 1)
  predictions = tf.reshape(output_layer, [-1])

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
