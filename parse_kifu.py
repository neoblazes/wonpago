# Play Go from Kifu.

from collections import defaultdict
import glob
import sys
import time

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
    print('Skips invalid move')
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
        print('Capture %d stones from (%d, %d)' % (len(group), pos[0], pos[1]))
        CaptureGroup(board, group)
        capture_count = capture_count + len(group)
        capture_pos = pos
  
  # Check Ko
  if capture_count == 1:
    group = set()
    GetConnented(board, group, x, y)
    if len(group) == 1 and GetFreedom(board, group) == 1:
      return (capture_pos[0], capture_pos[1])

def GetWinner(summary, susun):
  if summary.find('RE[B') > 0:
    return 'B'
  elif summary.find('RE[W') > 0:
    return 'W'
  elif len(susun) > 1 and susun[-1][-2:] != '[]':
    return susun[-1][0]
  return 'J'

def InitBoard():
  board = [x[:] for x in [[' '] * 11] * 11]
  for i in range(11):
    board[i][0] = 'E'
    board[0][i] = 'E'
    board[i][10] = 'E'
    board[10][i] = 'E'
  return board

if len(sys.argv) == 1:
  print('Usage: python parer_kifu.py <file_pattern>')
  exit(1)

files = glob.glob(sys.argv[1])
win_count = defaultdict(lambda: 0)
for file in files:
  f = open(file)
  summary = f.readline()
  p1 = f.readline()
  p2 = f.readline()
  susun = f.readline()[1:-1].split(';')

  win = GetWinner(summary, susun)
  print(win, ': ', file)
  win_count[win] = win_count[win] + 1
  
  board = InitBoard()
  for move in susun:
    print(move)
    ko = PlayGo(board, move)
    if ko != None:
      print('Ko occured')
      board[ko[0]][ko[1]] = 'K'
      for row in board:
        print(row)
      board[ko[0]][ko[1]] = ' '
    else:
      for row in board:
        print(row)

print('win_count: ', win_count)