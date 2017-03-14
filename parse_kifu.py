# Play Go from Kifu.
# Sample usages:
# python parse_kifu.py small/* > small.csv
# python parse_kifu.py small/* B > small_black.csv
# python parse_kifu.py small/* W > small_white.csv

import glob
import logging
import sys

from collections import defaultdict

import play_go

SKIP_SEQUENCE = 5  # Skips too early stages

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

# Main code.
if len(sys.argv) == 1:
  print('Usage: python parer_kifu.py <file_pattern> [B/W]')
  exit(1)
SKIP_STONE = None
feature_fn = play_go.ToFeature
if len(sys.argv) > 2:
  feature_fn = play_go.ToFeatureWithLiberty
  SKIP_STONE = 1 - ENCODE[sys.argv[2]]

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
    if seq_cnt > SKIP_SEQUENCE and valid and ENCODE[move[0]] != SKIP_STONE:
      # Assumes that the win rate increases linearly. 0.1 from first move and 1 on 90% move.
      discount = min(1, seq_cnt / (len(sequence) * 1.0) + 0.1)
      # For easier training, encode black stone to positive number and negative for white one.
      # Also by adding 1, makes the result to float. 0=white win, 1=jigo and 2=black win.      
      feature = feature_fn(board, ENCODE[move[0]], ko, discount * ENCODE[result])
      print(','.join(list(map(str, feature))))
    seq_cnt = seq_cnt + 1
logging.warning('win_count: %s' % str(win_count))
