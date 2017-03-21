# Model for CNN estimator

import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

def model_fn(features, targets, mode, params):
  # Features are 13 of 9x9
  # Note that turn and ko info are set to valid_move map.
  board = tf.reshape(features, [-1, 9, 9, 13])  # x*y*f is correct?
  # No relu, input includes negative. 13x25x128 = 41600
  conv_in = tf.layers.conv2d(inputs=board, filters=128, kernel_size=[5, 5],
      padding="same")
  conv1 = tf.layers.conv2d(inputs=conv_in, filters=64, kernel_size=[3, 3],
      padding="same") # 128*9*64 = 73728
  conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[3, 3],
      padding="same") # 64*9*64 = 36864
  conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[3, 3],
      padding="same") # 64*9*64 = 36864
  conv4 = tf.layers.conv2d(inputs=conv3, filters=64, kernel_size=[3, 3],
      padding="same") # 64*9*64 = 36864
  conv_out = tf.layers.conv2d(inputs=conv4, filters=1, kernel_size=[1, 1],
      padding="same") # To reduce size
  # Flattens conv2d output and use it as move probility.
  conv_flat = tf.reshape(conv_out, [-1, 9 * 9 * 1])

  # Dense layer and output.
  # 81*128 = 10368
  dense = tf.layers.dense(inputs=conv_flat, units=128)
  # 128*83 = 10624
  logits = tf.layers.dense(inputs=dense, units=83) # pass + 81 position + surrender

  predictions = {
    "action": tf.argmax(input=logits, axis=1),
    "probabilities": tf.nn.softmax(logits)
  }

  # For predict mode.
  if targets == None:
    return model_fn_lib.ModelFnOps(
      mode=mode,
      predictions=predictions)

  # For modes with targets.
  loss = tf.losses.softmax_cross_entropy(targets, logits)

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
