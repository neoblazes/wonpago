# Training DNN classifier from Kifu.
# It saaumes that there are training.csv and test.csv files alreay esixting.
#
# Usage: python train_kifu.py [dir] [steps] [flip,H|V|A]
# Sample usage: python train_kifu.py train2000 2000

import numpy as np
import tensorflow as tf

import logging
import sys

MODEL_DIR=None
STEPS=200
FLIP=''
if len(sys.argv) >= 3:
  MODEL_DIR=sys.argv[1]
  print('Working on directory: ', MODEL_DIR)
  STEPS=int(sys.argv[2])
if len(sys.argv) >= 4:
  FLIP=sys.argv[3]


def flip_vertical(feature):
  for i in range(4):
    feature[i*9:i*9+9], feature[81-i*9-9:81-i*9] = feature[81-i*9-9:81-i*9], feature[i*9:i*9+9]

def flip_horizontal(feature):
  for i in range(9):
    feature[i*9+0:i*9+9] = np.flipud(feature[i*9+0:i*9+9])

print('Training %d steps' % STEPS)
logging.getLogger().setLevel(logging.INFO)

# Load and define features
training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename='large.csv', target_dtype=np.int, features_dtype=np.float32, target_column=-1)
test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename='test.csv', target_dtype=np.int, features_dtype=np.float32, target_column=-1)
x_train, x_test, y_train, y_test = training_set.data, test_set.data, \
  training_set.target, test_set.target
if FLIP == 'V' or FLIP == 'A':
  for feature in x_train:
    flip_vertical(feature)
if FLIP == 'H' or FLIP == 'A':
  for feature in x_train:
    flip_horizontal(feature)
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=84)]

# Training
config = tf.contrib.learn.RunConfig()
config.tf_config.gpu_options.allow_growth=True
regressor = tf.contrib.learn.DNNRegressor(
    model_dir=MODEL_DIR, config=config,
    feature_columns=feature_columns, hidden_units=[81, 81, 49, 25])
regressor.fit(x=x_train, y=y_train, steps=STEPS)

# Evalutes accuracy
results = regressor.evaluate(x=x_test, y=y_test, steps=1)
for key in sorted(results):
  print("%s: %s" % (key, results[key]))

# For DNNClassifier
#accuracy_score = classifier.evaluate(x=x_test, y=y_test)["accuracy"]
#print('Accuracy: {0:f}'.format(accuracy_score))

# print out predicts
#ds_predict_tf  = classifier.predict(x_test) 
#for prd in ds_predict_tf:
#  print(prd)
