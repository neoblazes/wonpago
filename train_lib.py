# Library for training

import numpy as np
import sys
import play_go

def get_dominions(black_stones, white_stones, black_base, white_base):
  black_padded = np.pad(black_base, 1, 'constant', constant_values=0)
  white_padded = np.pad(white_base, 1, 'constant', constant_values=0)
  black_expand = (np.roll(black_padded, 1) + np.roll(black_padded, -1) +
                  np.roll(black_padded, 1, 0) + np.roll(black_padded, -1, 0))[1:-1, 1:-1]
  white_expand = (np.roll(white_padded, 1) + np.roll(white_padded, -1) +
                  np.roll(white_padded, 1, 0) + np.roll(white_padded, -1, 0))[1:-1, 1:-1]
  black_dominion = black_stones + np.all([black_expand > 0, white_stones == 0, white_expand == 0], axis=0)
  white_dominion = white_stones + np.all([black_expand == 0, white_stones == 0, white_expand > 0], axis=0)
  black_dominion[black_dominion > 0] = 1  # nomalize to 1
  white_dominion[white_dominion > 0] = 1
  return black_dominion.astype(int), white_dominion.astype(int)

def parse_row(row, produce_dominion=False):
  board_exp = [[0] * 81 for _ in range(3)]
  liberty_idx = 81
  liberty_map = [[0] * 81 for _ in range(2)]
  group_idx = 81 * 2
  group_size = [[0] * 81 for _ in range(2)]
  valid_idx = 81 * 3
  valid_move_exp = [[0] * 81 for _ in range(2)]

  for i in range(81):
    stone = int(row[i])
    # Mark empty, black and white for each layer
    board_exp[stone][i] = 1
    if stone != 0:
      liberty_map[stone - 1][i] = row[liberty_idx + i]
      group_size[stone - 1][i] = row[group_idx + i]
  for i in range(81):
    valid_move = int(row[valid_idx + i])
    valid_move_exp[valid_move - 1][i] = 1
  
  x_out = []
  for l in [board_exp[0], board_exp[1], board_exp[2], liberty_map[0], liberty_map[1],
            group_size[0], group_size[1], valid_move_exp[0] + valid_move_exp[1]]: # 3 + 2 + 2 + 2 (9 layers)
    x_out += l

  if produce_dominion:    
    # Build dominion map and add 4 layers (2*2), 13 layers in total.
    # Adding direct dominion
    black_stones = np.array(board_exp[1]).reshape((9, 9))
    white_stones = np.array(board_exp[2]).reshape((9, 9))
    black_dominion, white_dominion = get_dominions(
      black_stones, white_stones, black_stones, white_stones)
    x_out += black_dominion.reshape((9*9)).tolist()
    x_out += white_dominion.reshape((9*9)).tolist()

    # Secondary dominion
    black_dominion, white_dominion = get_dominions(
    black_stones, white_stones, black_dominion, white_dominion)
    x_out += black_dominion.reshape((9*9)).tolist()
    x_out += white_dominion.reshape((9*9)).tolist()
  
  # Parse target (move)
  y_num = int(row[-2])
  return x_out, y_num

def target_nparray(target):
  npas = []
  for t in target:
    npa = [0] * 83
    if t == 0:
      npa[0] = 1
    elif t == 1:
      npa[82] = 1
    else:
      npa[play_go.PackAction(t)] = 1  # Convert 11~99 to 1~81
    npas.append(np.asarray(npa, dtype=np.float32))
  return np.array(npas) 

# Flip functions.
def flip_vertical(feature, target):
  for i in range(len(feature)):
    feature[i] = feature[i].reshape((-1, 9, 9))[:,::-1,:].reshape((-1))
    if target[i] < 11:
      continue
    y = int((target[i]) / 10)
    x = target[i] % 10
    y = 10 - y
    target[i] = y * 10 + x

def flip_horizontal(feature, target):
  for i in range(len(feature)):
    feature[i] = feature[i].reshape((-1, 9, 9))[:,:,::-1].reshape((-1))
    if target[i] == 0 or target[i] == 82:
      continue
    y = int((target[i]) / 10)
    x = target[i] % 10
    x = (10 - x)
    target[i] = y * 10 + x

def rot90(feature, target):
  for i in range(len(feature)):
    feature[i] = np.rot90(feature[i].reshape((-1, 9, 9)), 1, axes=(2,1)).reshape((-1))
    if target[i] == 0 or target[i] == 82:
      continue
    y = int((target[i]) / 10)
    x = target[i] % 10    
    target[i] = x * 10 + (10 - y)  # (x, y) -> (10 - y, x)
