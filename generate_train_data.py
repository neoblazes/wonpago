# Generates training data from rich feature csv.
# As feature getting larger, it takes 30 minutes to initialize.
#
# Usage: python generate_train_data.py <in_csv>

import numpy as np

import csv
import sys

from tensorflow.python.platform import gfile

import train_lib

in_csv=sys.argv[1]

with gfile.Open(in_csv) as inf:
  data_file = csv.reader(inf)
  data, target = [], []
  for row in data_file:
    x, y = train_lib.parse_row(row, True)
    print(','.join(list(map(str, x + [y]))))
