# Play Go from Kifu.
# Sample usage: python parse_kifu.py small/* > small.csv

import glob
import logging
import sys

from collections import defaultdict

import play_go

SKIP_SEQUENCE = -1  # Evaluates from first move with discounted result. (0.1~)

def GetWinner(summary, sequence):
  if summary.find('RE[B') > 0:
    return 'B'
  elif summary.find('RE[W') > 0:
    return 'W'
  elif len(sequence) > 1 and sequence[-1][-2:] != '[]':
    return sequence[-1][0]
  return 'J'

ENCODE = {'B': 1, 'W': -1, ' ': 0, 'J': 0}
def PlayGo(board, move):
  if len(move) < 5:
    logging.info('Skips invalid move')
    return False, None
  stone = move[0]
  x = ord(move[2]) - ord('a') + 1
  y = ord(move[3]) - ord('a') + 1
  return play_go.PlayGo(board, ENCODE[stone], x, y)

def GetBlackTerritory(summary, sequence):
  pos = summary.find('RE[')
  if pos > 0:
    ret = 44
    winner = summary[pos + 3]
    diff = float(summary[pos + 5:summary.find(']', pos)])
    if winner == 'B':
      ret = ret + diff
    else:
      ret = ret - diff    
    return int(ret + 0.5)  # Adjust komi to 7.
  elif len(sequence) > 1 and sequence[-1][-2:] != '[]':
    return None  # Skips surrender games.
  return 44  # 44 for Jigo

# Main code.
if len(sys.argv) == 1:
  print('Usage: python parer_kifu.py <file_pattern> [liberty?y]')
  exit(1)
feature_fn = play_go.ToFeature
if len(sys.argv) > 2:
  feature_fn = play_go.ToFeatureWithLiberty

win_count = defaultdict(lambda: 0)
for file in glob.glob(sys.argv[1]):
  f = open(file)  
  summary = f.readline()
  p1 = f.readline()
  p2 = f.readline()
  sequence = f.readline()[1:-1].split(';')
  if len(sequence) <= SKIP_SEQUENCE:
    logging.warning('Drops too short game')
    continue

  result = GetWinner(summary, sequence)
  #black_territory = GetBlackTerritory(summary, sequence)
  #if black_territory == None:
    # Skipps surrender games.
  #  continue
  win_count[result] = win_count[result] + 1
  
  board = play_go.InitBoard()
  seq_cnt = 0
  for move in sequence:
    valid, ko = PlayGo(board, move)
    if seq_cnt > SKIP_SEQUENCE and valid:
      # Assumes that the win rate increases linearly. 0.1 from first move and 1 on 90% move.
      discount = min(1, seq_cnt / (len(sequence) * 1.0) + 0.1)
      # For easier training, encode black stone to positive number and negative for white one.
      # Also by adding 1, makes the result to float. 0=white win, 1=jigo and 2=black win.
      feature = feature_fn(board, ENCODE[move[0]], ko, discount * ENCODE[result])
      print(','.join(list(map(str, feature))))
    seq_cnt = seq_cnt + 1
logging.warning('win_count: %s' % str(win_count))
