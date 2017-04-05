# RPC server for policy network.
#
# Sample usage: python policy_rpc.py policy_0322_f13 (or policy_0324_inception_res)
#
# The feature is board [0,1,2] * 81 + ko pos (11~99) + turn [1,2]
# board 0: empty, 1: black, 2: white, serialized by 11, 12, 13 ... 98, 99
# ko pos 91: (1, 9), 19: (9, 1)
# turn 1: black, 2: white
#
# Sample RPC:
# import xmlrpclib
# s = xmlrpclib.ServerProxy('http://localhost:11001')
# ret = s.get_forwards('0,' * 81 + '0,1')
# ret = s.get_forwards(['0,' * 81 + '0,1', '0,' * 81 + '0,2'])

import numpy as np
import tensorflow as tf

import importlib
import random
import sys

import play_go
import train_lib
import xmlrpclib

from SimpleXMLRPCServer import SimpleXMLRPCServer

if len(sys.argv) < 2:
  print('Usage: python policy_rpc.py <model_dir>')
  exit(1)
model_dir = sys.argv[1]

ACTION_CODE = { 0: 'P', 82: 'S' }

# Load model and predict
model_fn = importlib.import_module('%s.model_fn' % model_dir)
estimator = model_fn.GetEstimator(model_dir)

board, ko, turn = play_go.InitBoard()

def new():
  global board, ko, turn
  board, ko, turn = play_go.InitBoard()
  return 'OK'

def play(action):
  global board, ko, turn
  if action == 0:
    pass
  elif action == ko:
    return '?'
  else:
    valid, ko = play_go.PlayGo(board, turn, action)
  turn = play_go.FlipTurn(turn)
  return '='

def genmove():
  global board, ko, turn
  feature = play_go.ToFeature(board, ko, turn, 0, 0)
  return get_forwards(','.join(str(x) for x in feature[:-2]))

def show():
  global board, ko, turn
  return [l[1:-1] for l in board[1:-1]]

def get_forwards(feature_csvs):
  if not type(feature_csvs) is list:
    feature_csvs = [feature_csvs]
  x_tests = []
  for feature_csv in feature_csvs:
    feature = [ int(x) for x in feature_csv.split(',') ]
    board, ko, turn = play_go.FromFeature(feature + [0])
    feature = play_go.ToFeature(board, ko, turn, 0, 0, True, True)
    x_test, _ = train_lib.parse_row(feature, True)
    x_tests.append(np.asarray(x_test, dtype=np.float32))
  predicts = estimator.predict(np.array(x_tests))
  rets = []
  for predict in predicts:
    probabilities = list(predict['probabilities'])
    actions = []
    for i in range(len(probabilities)):
      actions.append([train_lib.UnpackAction(i) if i > 0 and i < 82
                      else ACTION_CODE[i], float(probabilities[i])])
    rets.append(sorted(actions, key=lambda x:x[1], reverse = True)[:10])
  if len(rets) == 1:
    return rets[0]
  return rets
print(get_forwards(['1,1,1,0,0,0,0,0,0,' * 9 + '0,1', '0,' * 81 + '0,1']))

server = SimpleXMLRPCServer(("", 11001))
print "Listening on port 11001..."
print "Supporting new(), play(<action>), genmove() "
print "and get_forwards() with both of single feature csv or list of csv string."
server.register_function(new, "new")
server.register_function(play, "play")
server.register_function(genmove, "genmove")
server.register_function(show, "show")
server.register_function(get_forwards, "get_forwards")
server.serve_forever()
