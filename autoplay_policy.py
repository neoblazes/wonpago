# Auto play with a model.
#
# Sample usage: python autoplay_policy.py policy_0322_f13 1

import numpy as np
import tensorflow as tf

import importlib
import random
import subprocess
import sys

import play_go
import train_lib

from datetime import datetime

if len(sys.argv) < 2:
  print('Usage: python autoplay_policy.py <model_dir> [epsilon]')
  print('  epsilon 0: always play at 1, epsilon 1: follow softmax prob')
  print('  Note that sum of actions is smaller than 1 by valid move filtering.')
  print('  So epsilon 0.8 will be similar to softmax prob than 1')
  exit(1)
model_dir = sys.argv[1]
epsilon = 0.0
if len(sys.argv) == 3:
  epsilon = float(sys.argv[2])

STONE_CODE = ' BW'
POS_CODE = ' abcdefghi'
ACTION_CODE = { 0: 'P', 82: 'S' }

move_sequences = []

# Load model and predict
model_fn = importlib.import_module('%s.model_fn' % model_dir)
estimator = model_fn.GetEstimator(model_dir)
board, ko, turn = play_go.InitBoard()
passed = False
while True:
  feature = play_go.ToFeature(board, ko, turn, 0, 0, True, True)
  print(play_go.SPrintBoard(feature[:-1]))
  x_test, _ = train_lib.parse_row(feature, True)  # TODO: make configuable
  predict = estimator.predict(np.asarray(x_test, dtype=np.float32))
  probabilities = list(predict)[0]['probabilities']

  actions = []
  for i in range(len(probabilities)):
    actions.append([train_lib.UnpackAction(i) if i > 0 and i < 82
                    else ACTION_CODE[i], probabilities[i]])

  actions[0][1] = actions[0][1] / 5  # suppress pass
  sorted_actions = sorted(actions, key=lambda x:x[1], reverse = True)
  print(sorted_actions[:5])
  if sorted_actions[0][0] == 'P':
    move_sequences.append('%s[]' % STONE_CODE[turn])
    ko = 0
    if passed:
      print('Both passed')
      break
    print('Passed')
    passed = True
    turn = play_go.FlipTurn(turn)
    continue
  rand = random.random() * epsilon
  for a in sorted_actions:
    action = a[0]
    if action == 'S':
      print('Surrender signal, but will continue for test')
      continue
    if action == 'P':
      continue
    if rand > a[1]:
      rand -= a[1]
      continue
    # TODO: Should should check Ko in PlayGo()
    if action == ko:
      continue
    valid, ko = play_go.PlayGo(board, turn, action)
    if valid:
      move_sequences.append('%s[%s%s]' % (
              STONE_CODE[turn], POS_CODE[action % 10], POS_CODE[action / 10]))
      passed = False
      break
  turn = play_go.FlipTurn(turn)

# Stores sgf to autoplay.sgf file.
if len(move_sequences) > 2:  # Do not save if no sequences.
  sgf_str = ('(;GM[1]SZ[9]KM[7]RU[Chinese]\n'
             'PB[wonpago_policy_b]PW[wonpago_policy_w]\n'
             ';%s)' % ';'.join(move_sequences[:-2]))
  with open('autoplay.sgf', 'w') as f:
    f.write(sgf_str)

# Fast rollouts by using gnugo.
timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
p = subprocess.Popen(['gnugo', '-l', 'autoplay.sgf', '-o', 'autoplay/%s.sgf' % timestamp,
                      '--score', 'finish', '--chinese-rules'],
                     stdout=subprocess.PIPE)
out, err = p.communicate()
result_code = out.split('\n')[-2][0]
print out
print 'Result code: %s' % result_code
