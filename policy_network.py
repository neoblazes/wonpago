# Show best move candidates by evaluating all next moves.

import numpy as np
import tensorflow as tf

import copy
import sys

from collections import defaultdict

if len(sys.argv) < 2:
  print('Usage: python test_predict.py <model_dir>')
  exit(1)
model_dir = sys.argv[1]


def NearPositions(x, y):
  return ((x-1, y), (x+1, y), (x, y-1), (x, y+1))

def GetConnented(board, group, x, y):
  group.add((x, y))
  stone = board[x][y]
  for pos in NearPositions(x, y):
    if board[pos[0]][pos[1]] == stone and not (pos[0], pos[1]) in group:
      GetConnented(board, group, pos[0], pos[1])
      
def GetFreedom(board, group):
  # It should check dup of freedom. Just ok to check 0 or Ko.
  freedom = 0
  for pos in group:
    for n_pos in NearPositions(pos[0], pos[1]):
      if board[n_pos[0]][n_pos[1]] == ' ':
        freedom = freedom + 1
  return freedom

def CaptureGroup(board, group):
  for pos in group:
    board[pos[0]][pos[1]] = ' '

def IsOpponentStone(target, source):
  return target in ('B', 'W') and target != source

def PlayGo(board, stone, x, y):
  board[x][y] = stone
  
  # Capture stones
  capture_count = 0
  capture_pos = None
  for pos in NearPositions(x, y):
    if IsOpponentStone(board[pos[0]][pos[1]], stone):
      group = set()
      GetConnented(board, group, pos[0], pos[1])
      freedom = GetFreedom(board, group)      
      if freedom == 0:
        CaptureGroup(board, group)
        capture_count = capture_count + len(group)
        capture_pos = pos

  # Check forbidden move
  group = set()
  GetConnented(board, group, x, y)
  if GetFreedom(board, group) == 0:
    return False

  # Check Ko
  if capture_count == 1:
    if len(group) == 1 and GetFreedom(board, group) == 1:
      return (capture_pos[0], capture_pos[1])

def InitBoard():
  board = [x[:] for x in [[' '] * 11] * 11]
  for i in range(11):
    board[i][0] = 'E'
    board[0][i] = 'E'
    board[i][10] = 'E'
    board[10][i] = 'E'
  return board

# Load model and predict
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=84)]
regressor = tf.contrib.learn.DNNRegressor(
    model_dir=model_dir,
    feature_columns=feature_columns, hidden_units=[81, 81, 49, 25])

def GetPredict(feature):
  x_test = np.array([feature], dtype=float)
  return list(regressor.predict(x_test))[0]

# Print out human readable.
BOARD_CHAR = { -1: 'O', 1: '@', 0: '.', '2': '?' }
TURN_MSG = { 1: 'BLACK(@)', -1: 'WHITE(O)' }
RESULT_MSG = { 0: 'WHITE', 1: 'JIGO', 2: 'BLACK' }
def PrintBoard(feature, pred):
  board = feature[:81]
  last_move = feature[81]
  ko_pos = feature[82:84]
  pos = 0
  for row in range(1, 10):
    outstr = ''
    for col in range(1, 10):
      if row == ko_pos[0] and col == ko_pos[1]:
        outstr = outstr + '*'
      else:
        outstr = outstr + BOARD_CHAR[board[pos]]
      pos = pos + 1
    print(outstr)
  print('Last move %s, predict(W(-1)~B(1)): %f\n' %
        (TURN_MSG[last_move], pred - 1))

# Main loop
while True:
  feature = input('Enter board feature [[board]*81, last_move, [ko]*2]:\n')
  if len(feature) >= 84:
    feature = list(feature)
  else:
    feature = feature.split(',')
  if len(feature) < 84:
    continue
  feature = list(map(int, feature))[:84]
  PrintBoard(feature, GetPredict(feature))
  board = InitBoard()
  ko_x = int(feature[82])
  ko_y = int(feature[83])
  next_move = 0 - int(feature[81])
  stone = 'B'
  if feature[81] == 1:
    stone = 'W'
  idx = 0
  for i in range(1,10):
    for j in range(1,10):
      if feature[idx] == -1:
        board[i][j] = 'W'
      elif feature[idx] == 1:
        board[i][j] = 'B'
      idx = idx + 1

  features = []
  move_scores = {}  
  # Try all valid moves
  for i in range(1,10):
    for j in range(1,10):
      if not board[i][j] == ' ' or (i == ko_x and j == ko_y):
        continue
      board2 = copy.deepcopy(board)  # make clone for a move
      ret = PlayGo(board2, stone, i, j)
      if ret == False:
        continue
      feature2 = list(feature)  # make clone for eval
      feature2[(i-1)*9+j-1] = next_move
      feature2[81] = next_move
      if ret != None:
        feature2[82] = ret[0]
        feature2[83] = ret[1]
      # Push to eval
      features.append(feature2)
  # Batch eval (for performance)
  x_test = np.array(features, dtype=float)
  pred_tf = regressor.predict(x_test)
  idx = 0
  for pred in pred_tf:
    move_scores[','.join(list(map(str, features[idx])))] = pred
    idx = idx + 1    
  for k, v in sorted(move_scores.items(), key =lambda x:x[1], reverse = (stone=='B'))[:10]:
    PrintBoard(list(map(int, k.split(','))), v)
