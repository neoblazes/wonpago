import sys
import xmlrpclib

s = xmlrpclib.ServerProxy('http://neochoi2.seo.corp.google.com:11001/')

command_list = set(['clear_board', 'known_command', 'genmove', 'play'])
turn = 1
line = sys.stdin.readline()
while line:
  tokens = line.replace('\n', '').split(' ')
  if tokens[0] == 'known_command':
    print(command_list)
  elif tokens[0] == 'clear_board':
    s.new()
    turn = 1
    print('=')
  elif tokens[0] == 'play':
    if ((tokens[1] == 'black' and turn == 2) or
        (tokens[1] == 'white' and turn == 1)):
      s.play(0)  # pass once to flip turn
      turn = 2 - turn
    pos = (ord(tokens[2][0]) - ord('a') + 1) * 10 + int(tokens[2][1])
    s.play(pos)
    print('=')
  elif tokens[0] == 'genmove':
    actions = s.genmove()
    print(actions)
    for ap in actions:
      a = ap[0]
      if a != 'P' and a != 'S':
        print('= %s%d' % (chr(ord('a') + int(a / 10) - 1) , int(a) % 10))
        break
  line = sys.stdin.readline()
