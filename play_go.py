# Library to play Go.

import copy

def NearPositions(x, y):
  return [pos for pos in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
          if pos[0] > 0 and pos[0] < 10 and pos[1] > 0 and pos[1] < 10]

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
  group_map = [0] * 81
  idx = 0
  for i in range(1,10):
    for j in range(1,10):
      # Assumes that there is no liberty == 0 stone.
      if (board[i][j] == 1 or board[i][j] == -1) and liberty_map[idx] == 0:
        group = set()
        GetConnented(board, group, i, j)
        liberty = GetLiberty(board, group)
        for pos in group:
          group_map[(pos[0] - 1) * 9 + pos[1] - 1] = len(group)
          liberty_map[(pos[0] - 1) * 9 + pos[1] - 1] = liberty
      idx = idx + 1
  return liberty_map, group_map

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

def GetValidMoveMap(board, last_move, ko):
  ko_tuple = tuple(ko)
  moves = [0] * 81  
  next_move = -last_move
  idx = -1
  # Try all valid moves
  for i in range(1,10):
    for j in range(1,10):
      idx = idx + 1
      if not board[i][j] == 0 or (i, j) == ko_tuple:  # don't compare list
        continue
      board2 = copy.deepcopy(board)  # make clone for a move
      valid, ko = PlayGo(board2, next_move, i, j)
      if not valid:
        continue
      moves[idx] = next_move
  return moves

def FowardFeatures(feature):
  # TODO: feature and full_feature are confusing. Use status for small feature.
  board, last_move, ko = FromFeature(feature)
  ko_tuple = tuple(ko)
  next_move = -last_move
  features = []
  # Try all valid moves
  for i in range(1,10):
    for j in range(1,10):
      if not board[i][j] == 0 or (i, j) == ko_tuple:
        continue
      board2 = copy.deepcopy(board)  # make clone for a move
      valid, ko = PlayGo(board2, next_move, i, j)
      if not valid:
        continue      
      features.append(ToFeature(board2, next_move, ko, 0, True, True)[:-1])
  return features

def InitBoard():
  board = [x[:] for x in [[0] * 11] * 11]
  # TODO: merge board and feature
  return board

def FromFeature(feature):
  board = InitBoard()
  last_move = int(feature[-3])
  ko = [int(feature[-2]), int(feature[-1])]
  idx = 0
  for i in range(1,10):
    for j in range(1,10):
      if abs(feature[idx]) == 1:
        # 0.5 and -0.5 indicates ko. TODO: 0.5 is deprecated.
        board[i][j] = feature[idx]
      idx = idx + 1
  return board, last_move, ko

def ToFeature(board, last_move, ko, result, add_liberty=False, add_move=False):
  if ko == None:
    ko = [0, 0]
  board_serial = [item for innerlist in board[1:-1] for item in innerlist[1:-1]]
  if add_liberty:
    liberty_map, group_map = GetLibertyMap(board)
    board_serial = board_serial + liberty_map + group_map
  if add_move:
    board_serial = board_serial + GetValidMoveMap(board, last_move, ko)
  return board_serial + [last_move] + ko + [result]

# Print out human readable.
BOARD_CHAR = { -1: 'O', 1: '@', 0: '.' }
TURN_MSG = { 1: 'BLACK(@)', -1: 'WHITE(O)', 0: '?' }
def SPrintBoard(feature, detail=False):
  lines = []
  board = feature[:81]
  liberty = feature[81:81*2]
  group = feature[81*2:81*3]
  valid_move = feature[81*3:81*4]
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
    if detail:
      # Rich output
      lines.append('%s  %s %s %s' % (outstr,
                                     [int(l) for l in liberty[pos-9:pos]],
                                     [int(l) for l in group[pos-9:pos]],
                                     [int(l) for l in valid_move[pos-9:pos]]))
    else:
      lines.append(outstr)
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
    feature = list(map(float, feature))[:-1]
    board, last_move, ko = FromFeature(feature)
    print(SPrintBoard(ToFeature(board, last_move, ko, 0, True, True)[:-1], True))

if __name__ == "__main__":
    main()
