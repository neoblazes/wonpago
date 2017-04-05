# Model for CNN estimator

import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

def model_fn(features, targets, mode, params):
  # Features are 13 of 9x9
  # Note that turn and ko info are set to valid_move map.
  # Original input was (-1, 13, 9, 9), will be reshaped to (-1, 9, 9, 13)
  board = tf.transpose(tf.reshape(features, [-1, 13, 9, 9]), perm=[0, 2, 3, 1])
  conv1_55 = tf.layers.conv2d(inputs=board, filters=96, kernel_size=[5, 5],
      padding="same", activation=tf.nn.relu)  # 13*25*96 = 31200
  conv1_33 = tf.layers.conv2d(inputs=board, filters=256, kernel_size=[3, 3],
      padding="same", activation=tf.nn.relu)  # 13*9*256 = 29952
  inception1 = tf.layers.conv2d(inputs=tf.concat([conv1_55, conv1_33], -1),
      filters=64, kernel_size=[1, 1], padding="same")  # (96+256)*1*64 = 22528

  conv2_33_1_1 = tf.layers.conv2d(inputs=inception1, filters=32, kernel_size=[3, 3],
      padding="same", activation=tf.nn.relu)  # 64*9*32 = 18432
  conv2_33_1_2 = tf.layers.conv2d(inputs=conv2_33_1_1, filters=32, kernel_size=[3, 3],
      padding="same", activation=tf.nn.relu)  # 32*9*32 = 9216
  conv2_33_2 = tf.layers.conv2d(inputs=inception1, filters=48, kernel_size=[3, 3],
      padding="same", activation=tf.nn.relu)  # 64*9*48 = 27648
  inception2 = tf.layers.conv2d(inputs=tf.concat([conv2_33_1_2, conv2_33_2, inception1], -1),
      filters=64, kernel_size=[1, 1], padding="same")  # (64+48+32)*1*64 = 9216

  conv3_33_1_1 = tf.layers.conv2d(inputs=inception2, filters=32, kernel_size=[3, 3],                      padding="same", activation=tf.nn.relu)  # 64*9*32 = 18432
  conv3_33_1_2 = tf.layers.conv2d(inputs=conv3_33_1_1, filters=32, kernel_size=[3, 3],                    padding="same", activation=tf.nn.relu)  # 32*9*32 = 9216
  conv3_33_2 = tf.layers.conv2d(inputs=inception2, filters=48, kernel_size=[3, 3],                        padding="same", activation=tf.nn.relu)  # 64*9*48 = 27648
  inception3 = tf.layers.conv2d(inputs=tf.concat([conv3_33_1_2, conv3_33_2, inception2], -1),
      filters=64, kernel_size=[1, 1], padding="same")  # (64+48+32)*1*64 = 9216

  conv4_33_1_1 = tf.layers.conv2d(inputs=inception3, filters=32, kernel_size=[3, 3],
      padding="same", activation=tf.nn.relu)  # 64*9*32 = 18432
  conv4_33_1_2 = tf.layers.conv2d(inputs=conv4_33_1_1, filters=32, kernel_size=[3, 3],
      padding="same", activation=tf.nn.relu)  # 32*9*32 = 9216
  conv4_33_2 = tf.layers.conv2d(inputs=inception3, filters=48, kernel_size=[3, 3],
      padding="same", activation=tf.nn.relu)  # 64*9*48 = 27648
  inception4 = tf.layers.conv2d(inputs=tf.concat([conv4_33_1_2, conv4_33_2, inception3], -1),
      filters=64, kernel_size=[1, 1], padding="same")  # (64+48+32)*1*64 = 9216

  conv_out = tf.layers.conv2d(inputs=tf.concat([inception4, inception2, inception3], -1),
      filters=2, kernel_size=[1, 1], padding="same")  # (64+64+64)*1*2 = 384
  valid_moves = tf.reshape(tf.reduce_max(
          tf.slice(board, [-1, 0, 0, 7], [-1, 9, 9, 9]), 3), [-1, 9, 9, 1])

  conv_flat = tf.reshape(tf.multiply(conv_out, valid_moves), [-1, 9 * 9 * 2])
  special_actions = tf.layers.dense(inputs=conv_flat, units=2)  # 81*2*2 = 324

  playing, _ = tf.split(conv_flat, [81, 81], axis=1)
  passing, surrender = tf.split(special_actions, [1, 1], axis=1)
  # logits: pass + 81 positions + surrender, total freedom is 260k (260036).
  logits = tf.concat([passing, playing, surrender], 1)

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
