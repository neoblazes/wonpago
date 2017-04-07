# Library to play Go.

import copy
import logging

BLACK = 1
WHITE = 2
PASS = 0
SURRENDER = 1

class PlayLog:
  def __init__(self, board, ko, turn, action, winner):
    """ Defines play log.
    board: [0/1/2] * 81
    ko: 1~81, 0 means no Ko.
    turn: black:1, white:2
    action: 0: pass, 11~99: play, 1: surrender
    result: 1: black, 1.5: jigo, 2: white
    """
    self.board = board
    self.ko = ko
    self.turn = turn
    self.action = action
    self.winner = winner

  def ToCsv(self):
    board_serial = [str(item) for innerlist in self.board[1:-1] for item in innerlist[1:-1]]
    return ','.join(board_serial) + ',%d,%d,%d,%d' % (self.ko, self.turn, self.action, self.winner)

  def ToFeature(self):
    # TODO: implement.
    return None

def NearPositions(x, y):
  return [pos for pos in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
          if pos[0] > 0 and pos[0] < 10 and pos[1] > 0 and pos[1] < 10]

def GetConnented(board, group, x, y):
  # TODO group = set()
  group.add((x, y))
  stone = board[y][x]
  for pos in NearPositions(x, y):
    if board[pos[1]][pos[0]] == stone and not (pos[0], pos[1]) in group:
      GetConnented(board, group, pos[0], pos[1])

def GetLibertySet(board, group):
  liberty = set()
  # It should check dup of liberty. Just ok to check 0 or Ko.
  for pos in group:
    for n_pos in NearPositions(pos[0], pos[1]):
      if board[n_pos[1]][n_pos[0]] == 0:
        liberty.add((n_pos[0], n_pos[1]))
  return liberty

def GetLiberty(board, group):
  return len(GetLibertySet(board, group))

def CaptureGroup(board, group):
  for pos in group:
    board[pos[1]][pos[0]] = 0

def IsEmpty(board, x, y):
  return board[y][x] == 0

def IsOpponentStone(target, source):
  return target in (BLACK, WHITE) and target != source

def GetLibertyMap(board):
  liberty_map = [0] * 81
  group_map = [0] * 81
  atari_pos = [[], []]
  # TODO: self_atari_map = [0] * 81
  idx = 0
  for i in range(1,10):
    for j in range(1,10):
      # Assumes that there is no liberty == 0 stone.
      if (board[i][j] == BLACK or board[i][j] == WHITE) and liberty_map[idx] == 0:
        group = set()
        GetConnented(board, group, j, i)
        liberty_set = GetLibertySet(board, group)
        for pos in group:
          group_map[(pos[1] - 1) * 9 + pos[0] - 1] = len(group)
          liberty_map[(pos[1] - 1) * 9 + pos[0] - 1] = len(liberty_set)
        if len(liberty_set) == 1:
          atari_pos[FlipTurn(board[i][j]) - 1].append(liberty_set.pop())
      idx = idx + 1
  return liberty_map, group_map # , atari_pos

def IsCapturing(board, turn, x, y):
  for pos in NearPositions(x, y):
    if IsOpponentStone(board[pos[1]][pos[0]], turn):
      group = set()
      GetConnented(board, group, pos[0], pos[1])
      liberty = GetLiberty(board, group)
      if liberty == 1:
        return True
  return False

def IsSelfAtari(board, turn, x, y):
  board2 = copy.deepcopy(board)  # make clone for probe
  board2[y][x] = turn
  group = set()
  GetConnented(board2, group, x, y)
  liberty = GetLiberty(board2, group)
  return liberty < 2  # True for both of self atari and self die.

def EncodePos(x, y):
  return y * 10 + x

def DecodePos(pos):
  return [pos % 10, int(pos / 10)]

def FlipTurn(turn):
  return 3 - turn

def PlayGo(board, turn, action):
  if action % 10 == 0:  # mod 0 is used for special actions
    return True, 0
  x, y = DecodePos(action)
  if board[y][x] != 0:
    return False, 0
  board[y][x] = turn

  # Capture stones
  capture_count = 0
  capture_pos = None
  for pos in NearPositions(x, y):
    if IsOpponentStone(board[pos[1]][pos[0]], turn):
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
    board[y][x] = 0
    return False, 0

  # Check Ko
  if capture_count == 1:
    if len(group) == 1 and GetLiberty(board, group) == 1:
      return True, EncodePos(capture_pos[0], capture_pos[1])
  return True, 0

def PlayGoXy(board, stone, x, y):
  return PlayGo(board, stone, EncodePos(x, y))

def GetValidMoveMap(board, ko, turn, liberty_map):
  ko_tuple = tuple(DecodePos(ko))
  moves = [0] * 81
  idx = -1
  # Try all valid moves
  for y in range(1,10):
    for x in range(1,10):
      idx = idx + 1
      if not board[y][x] == 0 or (x, y) == ko_tuple:  # don't compare list
        continue
      need_play = True
      # Checks self liberty.
      for pos in NearPositions(x, y):
        if (board[pos[1]][pos[0]] == 0 or
            (board[pos[1]][pos[0]] == turn and
             liberty_map[(pos[1]-1)*9+pos[0]-1] > 1)):
          need_play = False
      if need_play:
        board2 = copy.deepcopy(board)  # make clone for a move
        valid, ko = PlayGoXy(board2, turn, x, y)
        if not valid:
          if not need_play:
            logging.critical('Bug on liberty check, (%d, %d)', y , x)
          continue
      moves[idx] = turn
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
  return board, 0, 1

def FromFeature(feature):
  board, _, _ = InitBoard()
  ko = feature[-3]
  turn = feature[-2]
  idx = 0
  for i in range(1,10):
    for j in range(1,10):
      board[i][j] = feature[idx]
      idx = idx + 1
  return board, ko, turn

def ToFeature(board, ko, turn, action, result, add_liberty=False, add_move=False):
  board_serial = [item for innerlist in board[1:-1] for item in innerlist[1:-1]]
  liberty_map = None
  if add_liberty:
    liberty_map, group_map = GetLibertyMap(board)
    board_serial = board_serial + liberty_map + group_map
  if add_move:
    if liberty_map == None:
      liberty_map, _ = GetLibertyMap(board)
    board_serial = board_serial + GetValidMoveMap(board, ko, turn, liberty_map)
  return board_serial + [ko, turn, action, result]

# Print out human readable.
BOARD_CHAR = { 2: 'O', 1: 'X', 0: '.' }
TURN_MSG = { 1: 'BLACK(X)', 2: 'WHITE(O)', 0: '?' }
def SPrintBoard(feature, detail=False):
  lines = []
  board = feature[:81]
  liberty = feature[81:81*2]
  group = feature[81*2:81*3]
  valid_move = feature[81*3:81*4]
  ko = DecodePos(feature[-3])
  turn = feature[-2]
  pos = 0
  for row in range(1, 10):
    outstr = ''
    for col in range(1, 10):
      if row == ko[1] and col == ko[0]:
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
  lines.append('Last move %s' % TURN_MSG[FlipTurn(turn)])
  return '\n'.join(lines)


# Main loop
def main():
  while True:
    feature = input('Enter board feature [[board]*81 + ko + turn + action + result]:\n')
    if isinstance(feature, (list, tuple)):
      feature = list(feature)
    else:
      feature = feature.split(',')
    feature = list(map(int, feature))[:-1]
    if len(feature) > 81 * 2:
      print(SPrintBoard(feature, True))
    else:
      board, ko, turn = FromFeature(feature)
      print(SPrintBoard(ToFeature(board, ko, turn, 0, 0, True, True)[:-1], True))

if __name__ == "__main__":
    main()
