# Play Go from Kifu.
# Sample usage: python parse_kifu.py small/* small.log > small.csv

import glob
import logging
import sys
import time

from collections import defaultdict


SKIP_SEQUENCE = 10  # Only save after 10 sequences

def NearPositions(x, y):
  return ((x-1, y), (x+1, y), (x, y-1), (x, y+1))

def GetConnented(board, group, x, y):
  group.add((x, y))
  stone = board[x][y]
  for pos in NearPositions(x, y):
    if board[pos[0]][pos[1]] == stone and not (pos[0], pos[1]) in group:
      GetConnented(board, group, pos[0], pos[1])
      
def GetFreedom(board, group):
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

def PlayGo(board, move):
  if len(move) < 5:
    logging.info('Skips invalid move')
    return
  stone = move[0]
  x = ord(move[2]) - ord('a') + 1
  y = ord(move[3]) - ord('a') + 1
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
        logging.info('Capture %d stones from (%d, %d)' % (len(group), pos[0], pos[1]))
        CaptureGroup(board, group)
        capture_count = capture_count + len(group)
        capture_pos = pos
  
  # Check Ko
  if capture_count == 1:
    group = set()
    GetConnented(board, group, x, y)
    if len(group) == 1 and GetFreedom(board, group) == 1:
      return (capture_pos[0], capture_pos[1])

def GetWinner(summary, sequence):
  if summary.find('RE[B') > 0:
    return 'B'
  elif summary.find('RE[W') > 0:
    return 'W'
  elif len(sequence) > 1 and sequence[-1][-2:] != '[]':
    return sequence[-1][0]
  return 'J'

def InitBoard():
  board = [x[:] for x in [[' '] * 11] * 11]
  for i in range(11):
    board[i][0] = 'E'
    board[0][i] = 'E'
    board[i][10] = 'E'
    board[10][i] = 'E'
  return board

# For easier training, encode black stone to positive number and negative for white one.
# Also by adding 1, makes the result to unsigned int. 0=white win, 1=jigo and 2=black win.
ENCODE = {'B': 1, 'W': -1, ' ': 0, 'J': 0}
def ToCsv(board, last_move, ko, result):
  if ko == None:
    ko = (0, 0)
  board_serial = ','.join(str(ENCODE[item]) for innerlist in board[1:-1] for item in innerlist[1:-1])
  return ('%s,%d,%d,%d,%d' % (board_serial, ENCODE[last_move], ko[0], ko[1], ENCODE[result]))


# Main code.
if len(sys.argv) == 1:
  print('Usage: python parer_kifu.py <file_pattern> [log_file]')
  exit(1)

if len(sys.argv) > 2:
  logging.basicConfig(filename=sys.argv[2],level=logging.DEBUG)

win_count = defaultdict(lambda: 0)
for file in glob.glob(sys.argv[1]):
  logging.info('Starting new game: %s' % file)
  f = open(file)  
  summary = f.readline()
  p1 = f.readline()
  p2 = f.readline()
  sequence = f.readline()[1:-1].split(';')
  if len(sequence) <= SKIP_SEQUENCE:
    logging.warning('Drops too short game')
    continue

  result = GetWinner(summary, sequence)  
  win_count[result] = win_count[result] + 1
  
  board = InitBoard()
  seq_cnt = 0
  for move in sequence:
    logging.info(move)
    ko = PlayGo(board, move)
    if ko != None:
      logging.info('Ko occured')
      board[ko[0]][ko[1]] = 'K'
      for row in board[1:-1]:  # Skips edge on logging
        logging.info(row[1:-1])
      board[ko[0]][ko[1]] = ' '
    else:
      for row in board[1:-1]:
        logging.info(row[1:-1])
    if seq_cnt > SKIP_SEQUENCE:
      print(ToCsv(board, move[0], ko, result))
    else:
      seq_cnt = seq_cnt + 1
  logging.info('End of gmae, result: %s' % result)

logging.info('win_count: %s' % str(win_count))
