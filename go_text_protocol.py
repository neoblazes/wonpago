import random
import sys
import xmlrpclib

import play_go

s = xmlrpclib.ServerProxy('http://neochoi2.seo.corp.google.com:11001/')

command_list = set(['clear_board', 'list_commands', 'genmove', 'play',
                    'protocol_version', 'get_random_seed', 'showboard',
                    'boardsize', 'komi', 'name', 'version', 'score', 'cpu_time'])
TURN_STR = { 'b': 1, 'w': 2, 'black': 1, 'white': 2 }

board, ko, turn = play_go.InitBoard()
move_count = 0
line = sys.stdin.readline()
while line:
  tokens = line.replace('\n', '').lower().split(' ')
  if tokens[0] == 'protocol_version':
    print ('= 2')
    print
  elif tokens[0] == 'get_random_seed':
    print('= 0')
    print
  elif tokens[0] == 'boardsize':
    print('=')
    print
  elif tokens[0] == 'komi':
    print('=')
    print
  elif tokens[0] == 'name':
    print('= wonpago policy')
    print
  elif tokens[0] == 'version':
    print('= 1')
    print
  elif tokens[0] == 'list_commands':
    print('=')
    print(command_list)
    print
  elif tokens[0] == 'clear_board':
    board, ko, turn = play_go.InitBoard()
    move_count = 0
    print('=')
    print
  elif tokens[0] == 'play':
    turn = TURN_STR[tokens[1]]
    if tokens[2] != 'pass':
      pos0 = tokens[2][0].replace('j', 'i')
      pos = (ord(pos0) - ord('a') + 1) * 10 + int(tokens[2][1])
      valid, ko = play_go.PlayGo(board, turn, pos)
    turn = play_go.FlipTurn(turn)
    move_count += 1
    print('=')
    print
  elif tokens[0] == 'genmove':
    turn = TURN_STR[tokens[1]]
    feature = play_go.ToFeature(board, ko, turn)[:-2]
    actions = s.get_forwards(','.join(list(map(str, feature))))
    prob_sum = 0.0
    for ap in actions:
      a = ap[0]
      if a in ('P', 'S'):
        continue
      prob_sum += ap[1]
    if move_count > 10 and (actions[0][0] == 'P' or prob_sum == 0.0):
      print('= PASS')
    else:
      rand = random.random() * prob_sum
      for ap in actions:
        a = ap[0]
        if a in ('P', 'S'):
          continue
        if rand < ap[1]:
          pos = int(a)
          break
        else:
          rand -= ap[1]
      play_go.PlayGo(board, turn, pos)
      move = '= %s%d' % (chr(ord('a') + int(pos / 10) - 1) , int(pos) % 10)
      print(move.replace('i', 'j'))
    turn = play_go.FlipTurn(turn)
    move_count += 1
    print
  elif tokens[0] == 'showboard':
    print('=')
    print(play_go.SPrintBoardRaw(board, ko, turn))
    print
  elif tokens[0] == 'final_score':
    print('= ?')
    print
  elif tokens[0] == 'cpu_time':
    print('= 0')
    print
  elif tokens[0] == 'quit':
    print('=')
    print
    sys.stdout.flush()
    exit(1)
  else:
    pass

  sys.stdout.flush()

  line = sys.stdin.readline()
