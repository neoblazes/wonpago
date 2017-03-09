# Training DNN classifier from Kifu.
# It saaumes that there are training.csv and test.csv files alreay esixting.
#
# Usage: python train_kifu.py [dir] [steps]
# Sample usage: python train_kifu.py train2000 2000

import numpy as np
import tensorflow as tf

import logging
import sys

MODEL_DIR=None
STEPS=200
if len(sys.argv) >= 3:
  MODEL_DIR=sys.argv[1]
  print('Working on directory: ', MODEL_DIR)
  STEPS=int(sys.argv[2])
print('Training %d steps' % STEPS)
logging.getLogger().setLevel(logging.INFO)

# Load and define features
training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename='training.csv', target_dtype=np.int, features_dtype=np.float32, target_column=-1)
test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename='small.csv', target_dtype=np.int, features_dtype=np.float32, target_column=-1)
x_train, x_test, y_train, y_test = training_set.data, test_set.data, \
  training_set.target, test_set.target
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=84)]

# Training
classifier = tf.contrib.learn.DNNClassifier(
    model_dir=MODEL_DIR, feature_columns=feature_columns, hidden_units=[20, 20], n_classes=3)
classifier.fit(x=x_train, y=y_train, steps=STEPS)

# Evalutes accuracy
accuracy_score = classifier.evaluate(x=x_test, y=y_test)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

# print out predicts
#ds_predict_tf  = classifier.predict(x_test) 
#for prd in ds_predict_tf:
#  print(prd)
