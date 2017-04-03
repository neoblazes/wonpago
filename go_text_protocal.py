import sys
import xmlrpclib

s = xmlrpclib.ServerProxy('http://neochoi2.seo.corp.google.com:11001/')

command_list = set(['clear_board', 'known_command', 'genmove', 'play', 'showboard'])
turn = 1
line = sys.stdin.readline()
while line:
  tokens = line.replace('\n', '').split(' ')
  if tokens[0] == 'known_command':
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
    pos = (ord(tokens[2][0]) - ord('a') + 1) * 10 + int(tokens[2][1])
    s.play(pos)
    turn = 3 - turn
    print('=')
    print
  elif tokens[0] == 'genmove':
    if ((tokens[1] == 'black' and turn == 2) or
        (tokens[1] == 'white' and turn == 1)):
      print('pass')
      s.play(0)  # pass once to flip turn
      turn = 3 - turn
    actions = s.genmove()
    for ap in actions:
      a = ap[0]
      if a != 'P' and a != 'S':
        pos = a
        break;
    s.play(pos)
    turn = 3 - turn
    print('= %s%d' % (chr(ord('a') + int(pos / 10) - 1) , int(pos) % 10))
    print
  elif tokens[0] == 'showboard':
    for l in s.show():
      print l
  else:
    print('?')

  line = sys.stdin.readline()
