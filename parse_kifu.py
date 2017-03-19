# Play Go from Kifu.
# Sample usages:
# python parse_kifu.py small/* > small.csv
# python parse_kifu.py small/* simple > small_simple.csv

import glob
import logging
import sys

from collections import defaultdict

import play_go

SKIP_SEQUENCE = -1  # Skips too early stages
RICH_OUTPUT = len(sys.argv) == 2


def GetWinner(summary, sequence):
  if summary.find('RE[B') > 0:
    return 0
  elif summary.find('RE[W') > 0:
    return 2
  elif len(sequence) > 1 and sequence[-1][-2:] != '[]':
    if sequence[-1][0] == 'B':
      return 0
    else:
      return 2
  return 1

def WinBySurrender(winner, summary):
  if winner == 1:
    return False
  return not summary.find('RE[') > 0

ENCODE = {'B': 1, 'W': -1, ' ': 0, 'J': 0}
def PlayGo(board, move):
  if len(move) < 5:
    logging.info('Skips invalid move')
    return False, None
  stone = move[0]
  x = ord(move[2]) - ord('a') + 1
  y = ord(move[3]) - ord('a') + 1
  return play_go.PlayGoXy(board, ENCODE[stone], x, y)

def ParseMove(move):
  turn = 1
  if move[0] == 'W':
    turn = 2
  if len(move) < 5:
    return turn, play_go.PASS  # passed
  x = ord(move[2]) - ord('a') + 1
  y = ord(move[3]) - ord('a') + 1
  return turn, play_go.EncodePos(x, y)

# Main code.
if len(sys.argv) == 1:
  print('Usage: python parer_kifu.py <file_pattern> [output_simple]')
  exit(1)

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

  winner = GetWinner(summary, sequence)
  surrendered = WinBySurrender(winner, summary)
  win_count[winner] = win_count[winner] + 1
  
  board, ko, turn = play_go.InitBoard()
  seq_cnt = 0
  for move in sequence:
    # Print before play.
    if len(move) == 0:
      continue
    turn, action = ParseMove(move)
    discount = min(1, seq_cnt / (len(sequence) * 1.0))
    feature = play_go.ToFeature(board, ko, turn, action, winner, RICH_OUTPUT, RICH_OUTPUT)
    print(','.join(list(map(str, feature))))
    
    valid, ko = play_go.PlayGo(board, turn, action)
    # TODO(neochio): fix encode.
    #if ko == None:
    #  ko = 0
    #else:
    #  ko = ko[0] * 9 + ko[1] - 9
    seq_cnt = seq_cnt + 1
  if surrendered:
    feature = play_go.ToFeature(board, ko, play_go.FlipTurn(turn), play_go.SURRENDER, winner, RICH_OUTPUT, RICH_OUTPUT)
    print(','.join(list(map(str, feature))))
logging.warning('win_count: %s' % str(win_count))
