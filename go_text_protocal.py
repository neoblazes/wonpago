import random
import sys
import xmlrpclib

s = xmlrpclib.ServerProxy('http://neochoi2.seo.corp.google.com:11001/')

command_list = set(['clear_board', 'list_commands', 'genmove', 'play',
                    'protocol_version', 'get_random_seed', 'showboard',
                    'boardsize', 'komi', 'name', 'version'])
turn = 1
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
    s.new()
    turn = 1
    print('=')
    print
  elif tokens[0] == 'play':
    if ((tokens[1] == 'black' and turn == 2) or
        (tokens[1] == 'white' and turn == 1)):
      s.play(0)  # pass once to flip turn
      turn = 3 - turn
    if tokens[2] != 'pass':
      pos0 = tokens[2][0].replace('j', 'i')
      pos = (ord(pos0) - ord('a') + 1) * 10 + int(tokens[2][1])
    s.play(pos)
    turn = 3 - turn
    print('=')
    print
  elif tokens[0] == 'genmove':
    if ((tokens[1] == 'black' and turn == 2) or
        (tokens[1] == 'white' and turn == 1)):
      s.play(0)  # pass once to flip turn
      turn = 3 - turn
    actions = s.genmove()
    prob_sum = 0.0
    for ap in actions:
      a = ap[0]
      if a in ('P', 'S'):
        continue
      prob_sum += ap[1]
    if actions[0][0] == 'P' or prob_sum == 0.0:
      print('= PASS')
    else:
      rand = random.random() * prob_sum
      for ap in actions:
        a = ap[0]
        if a in ('P', 'S'):
          continue
        if rand < ap[1]:
          pos = a
          break
        else:
          rand -= ap[1]
      s.play(pos)
      move = '= %s%d' % (chr(ord('a') + int(pos / 10) - 1) , int(pos) % 10)
      print(move.replace('i', 'j'))
    turn = 3 - turn
    print
  elif tokens[0] == 'showboard':
    print('=')
    for l in s.show():
      print l
    print
  else:
    print('?')

  sys.stdout.flush()

  line = sys.stdin.readline()
