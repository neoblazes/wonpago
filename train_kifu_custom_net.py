import argparse
import numpy
import random
import sys

import tensorflow as tf

NUM_HIDDEN = 3

def CsvToTensor(line):
  cols = line.strip().split(",")
  board_cols = []
  board_cols.append(cols[0:81])
  last_move_col = [[int(cols[81])] * NUM_HIDDEN]  # Why 2d?
  result = int(cols[84])
  result_col = [[0] * 3]
  result_col[0][result+1] = 1
  return board_cols, last_move_col, result_col

# Defines neural.
board = tf.placeholder(tf.float32, [None, 81])
#ko_status = tf.placeholder(tf.float32, [None, 81])
last_move = tf.placeholder(tf.float32)

W11 = tf.Variable(tf.zeros([81, NUM_HIDDEN]))
b11 = tf.Variable(tf.zeros([NUM_HIDDEN]))
#W12 = tf.Variable(tf.zeros([81, NUM_HIDDEN]))
#b12 = tf.Variable(tf.zeros([NUM_HIDDEN]))
W13 = tf.Variable(tf.zeros([NUM_HIDDEN]))
b13 = tf.Variable(tf.zeros([NUM_HIDDEN]))

hidden = tf.matmul(board, W11) + b11 + last_move * W13 + b13

W2 = tf.Variable(tf.zeros([NUM_HIDDEN, 3]))
b2 = tf.Variable(tf.zeros([3]))

predict = tf.nn.softmax(tf.matmul(hidden, W2) + b2)
result = tf.placeholder(tf.float32, [None, 3])

# define learner
cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=result, logits=predict))
train_step = tf.train.GradientDescentOptimizer(0.8).minimize(cross_entropy)


# Start session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train
inf = open('small.csv')
max_train = 10000
lines = inf.readlines()
inf.close()
for _ in range(max_train):
  idx = random.randrange(0, len(lines))
  board_cols, last_move_col, result_col = CsvToTensor(lines[idx])
  _, loss_val = sess.run([train_step, cross_entropy], feed_dict={board: board_cols, last_move: last_move_col, result: result_col})
  print('loss_val=%s' % loss_val)
  max_train = max_train - 1
  if max_train <= 0:
      break

# Eval
inf = open('small.csv')  # Use same file for now
max_eval = 1000
for line in inf:
  board_cols, last_move_col, result_col = CsvToTensor(line)
  _, res_predict = sess.run([train_step, predict], feed_dict={board: board_cols, last_move: last_move_col, result: result_col})

  print(board_cols)
  print('result=%s' % result_col)
  print('predict=%s' % res_predict)
  max_eval = max_eval - 1
  if max_eval <= 0:
    break
inf.close()
