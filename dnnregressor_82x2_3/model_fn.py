# Model for DNN regressor

import numpy as np
import tensorflow as tf

def GetEstimator(model_dir, config=None):
  config = tf.contrib.learn.RunConfig()
  config.tf_config.gpu_options.allow_growth=True
  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=84+81)]
  return tf.contrib.learn.DNNRegressor(
    model_dir=model_dir, config=config,
    feature_columns=feature_columns, hidden_units=[81, 81, 49, 25])
