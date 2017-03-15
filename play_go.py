# Library to play Go.

import copy

def NearPositions(x, y):
  return ((x-1, y), (x+1, y), (x, y-1), (x, y+1))

def GetConnented(board, group, x, y):
  group.add((x, y))
  stone = board[x][y]
  for pos in NearPositions(x, y):
    if board[pos[0]][pos[1]] == stone and not (pos[0], pos[1]) in group:
      GetConnented(board, group, pos[0], pos[1])

def GetLiberty(board, group):
  liberty = set()
  # It should check dup of liberty. Just ok to check 0 or Ko.
  for pos in group:
    for n_pos in NearPositions(pos[0], pos[1]):
      if board[n_pos[0]][n_pos[1]] == 0:
        liberty.add((n_pos[0], n_pos[1]))
  return len(liberty)

def CaptureGroup(board, group):
  for pos in group:
    board[pos[0]][pos[1]] = 0

def IsOpponentStone(target, source):
  return target in (1,-1) and target != source

def GetLibertyMap(board):
  liberty_map = [0] * 81
  idx = 0
  for i in range(1,10):
    for j in range(1,10):
      # Assumes that there is no liberty == 0 stone.
      if (board[i][j] == 1 or board[i][j] == -1) and liberty_map[idx] == 0:
        group = set()
        GetConnented(board, group, i, j)
        liberty = GetLiberty(board, group)
        for pos in group:
          liberty_map[(pos[0] - 1) * 9 + pos[1] - 1] = liberty
      idx = idx + 1
  return liberty_map

def PlayGo(board, stone, x, y):
  board[x][y] = stone

  # Capture stones
  capture_count = 0
  capture_pos = None
  for pos in NearPositions(x, y):
    if IsOpponentStone(board[pos[0]][pos[1]], stone):
      group = set()
      GetConnented(board, group, pos[0], pos[1])
      liberty = GetLiberty(board, group)
      if liberty == 0:
        CaptureGroup(board, group)
        capture_count = capture_count + len(group)
        capture_pos = pos

  # Check forbidden move
  group = set()
  GetConnented(board, group, x, y)
  if GetLiberty(board, group) == 0:
    return False, None

  # Check Ko
  if capture_count == 1:
    if len(group) == 1 and GetLiberty(board, group) == 1:
      return True, [capture_pos[0], capture_pos[1]]
  return True, None

def FowardFeatures(feature):
  board, last_move, ko = FromFeature(feature)
  next_move = -last_move
  features = []
  # Try all valid moves
  for i in range(1,10):
    for j in range(1,10):
      if not board[i][j] == 0 or [i, j] == ko:
        continue
      board2 = copy.deepcopy(board)  # make clone for a move
      valid, ko = PlayGo(board2, next_move, i, j)
      if not valid:
        continue      
      features.append(ToFeatureWithLiberty(board2, next_move, ko, 0)[:-1])
  return features

def InitBoard():
  board = [x[:] for x in [[0] * 11] * 11]
  for i in range(11):
    board[i][0] = 'E'
    board[0][i] = 'E'
    board[i][10] = 'E'
    board[10][i] = 'E'
  return board

def FromFeature(feature):
  board = InitBoard()
  last_move = int(feature[-3])
  ko = feature[-2:]
  idx = 0
  for i in range(1,10):
    for j in range(1,10):
      if abs(feature[idx]) == 1:
        # 0.5 and -0.5 indicates ko.
        board[i][j] = feature[idx]
      idx = idx + 1
  return board, last_move, ko

def ToFeature(board, last_move, ko, result):
  if ko == None:
    ko = [0, 0]
  else:
    board[ko[0]][ko[1]] = last_move / 2  # Set 0.5 or -0.5 for ko position.
  board_serial = [item for innerlist in board[1:-1] for item in innerlist[1:-1]]
  if not ko == None:
    board[ko[0]][ko[1]] = 0
  return board_serial + [last_move] + ko + [result]

def ToFeatureWithLiberty(board, last_move, ko, result):
  if ko == None:
    ko = [0, 0]
  else:
    board[ko[0]][ko[1]] = last_move / 2  # Set 0.5 or -0.5 for ko position.
  board_serial = [item for innerlist in board[1:-1] for item in innerlist[1:-1]]
  if not ko == None:
    board[ko[0]][ko[1]] = 0
  # Output 81*2 +4 for CNN model
  return board_serial + GetLibertyMap(board) + [last_move] + ko + [result]

def AttachLibertyToFeature(feature):
  board, last_move, ko = FromFeature(feature)
  return ToFeatureWithLiberty(board, last_move, ko, 0)

# Print out human readable.
BOARD_CHAR = { -1: 'O', 1: '@', 0: '.' }
TURN_MSG = { 1: 'BLACK(@)', -1: 'WHITE(O)', 0: '?' }
def SPrintBoard(feature):
  lines = []
  board = feature[:81]
  liberty = feature[81:81+81]
  last_move = feature[-3]
  ko = feature[-2:]
  pos = 0
  for row in range(1, 10):
    outstr = ''
    for col in range(1, 10):
      if row == ko[0] and col == ko[1]:
        outstr = outstr + '*'
      else:
        outstr = outstr + BOARD_CHAR[board[pos]]
      pos = pos + 1
    lines.append('%s  %s' % (outstr, [int(l) for l in liberty[pos-9:pos]]))
  lines.append('Last move %s' % TURN_MSG[int(last_move)])
  return '\n'.join(lines)


# Main loop
def main():
  while True:
    feature = input('Enter board feature [[board]*81, last_move, [ko]*2]:\n')
    if isinstance(feature, (list, tuple)):
      feature = list(feature)
    else:
      feature = feature.split(',')
    if len(feature) < 84:
      continue
    feature = list(map(float, feature))[:84]
    board, last_move, ko = FromFeature(feature)
    print(SPrintBoard(feature))

if __name__ == "__main__":
    main()
