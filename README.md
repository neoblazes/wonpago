The model is defined by [model_dir]/model_fn.py.

The network models use [board]*81 + [liberty] * 81 + last_move + ko_pos[2] as feature.
Note that the black stone is represented as '1', white stone as '-1' and empty space '0' on board.
ko_pos are actually not used by network, and the Ko position is marked as 0.5 or -0.5 on board, corresponding to whom made the Ko.

The prediction is value -1 ~ 1 where -1 means white win and 1 means black win, while 0 means jigo.

Sample execution:
0) Install TensorFlow (and CUDA), get the git files.
$ git clone https://github.com/neoblazes/wonpago

1) Unzip kifu file
$ unzip go9-large.zip -d go9-large

2) Generate features from kifu
$ python parse_kifu.py "go9-large/*" > large.csv

3) Train model
$ python train_kifu.py cnn_82x2_3_mini large.csv 1000

4) Test predict againt test.csv
$ python test_predict.py cnn_82x2_3_mini test.csv

5) Test predict for random opening positions
$ python opening_predict.py cnn_82x2_3_mini 3 10

6) Get recommended move for a feature
$ python eval_policy.py cnn_82x2_3_mini
...[give any of feature sample from csv files]
ex) 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,-1,0,-1,0,0,0,-1,1,1,-1,0,1,0,0,1,1,-1,1,-1,0,0,0,0,1,-1,-1,-1,1,1,0,0,0,-1,1,0,0,0,0,0,0,0,-1,0,-1,-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,4,0,5,0,3,0,0,0,1,4,4,5,0,3,0,0,3,3,2,4,5,0,0,0,0,3,2,2,2,4,4,0,0,0,4,2,0,0,0,0,0,0,0,4,0,5,5,3,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0.5901960784313726




Mar 15

Modulized all logic and make network model configuable per model_dir.
Establish CNN model with liberty map and turn info.

Mar 10

Implemented train_kifu.py
Used default DNNClassifer.

The training and test data set is built from kifu with timestamp
training: 2015-12-01~2015-12-25
test: 2015-12-26

And here is the run log with 20x20 hidden without CUDA.

The loss decrease 0.002 per 100 step.
100k step of the learning might show good progress.

>python train_kifu.py train2000 2000
...
INFO:tensorflow:Restoring parameters from train2000\model.ckpt-2
INFO:tensorflow:Saving checkpoints for 3 into train2000\model.ckpt.
INFO:tensorflow:step = 3, loss = 0.925718
INFO:tensorflow:global_step/sec: 0.825497
INFO:tensorflow:step = 103, loss = 0.880698 (664.161 sec)
INFO:tensorflow:Saving checkpoints for 104 into train2000\model.ckpt.
INFO:tensorflow:global_step/sec: 0.151344
INFO:tensorflow:step = 203, loss = 0.875539 (117.692 sec)
INFO:tensorflow:global_step/sec: 0.901925
INFO:tensorflow:step = 303, loss = 0.872281 (110.989 sec)
INFO:tensorflow:global_step/sec: 0.886161
INFO:tensorflow:step = 403, loss = 0.869591 (112.737 sec)
INFO:tensorflow:global_step/sec: 0.86491
INFO:tensorflow:step = 503, loss = 0.867261 (115.859 sec)
INFO:tensorflow:global_step/sec: 0.932845
INFO:tensorflow:step = 603, loss = 0.865196 (106.959 sec)
INFO:tensorflow:Saving checkpoints for 639 into train2000\model.ckpt.
INFO:tensorflow:global_step/sec: 0.902813
INFO:tensorflow:step = 703, loss = 0.863322 (110.774 sec)
INFO:tensorflow:global_step/sec: 0.875572
INFO:tensorflow:step = 803, loss = 0.861571 (114.457 sec)
INFO:tensorflow:global_step/sec: 0.90836
INFO:tensorflow:step = 903, loss = 0.859897 (384.756 sec)



Mar 9

Implemented parse_kifu.py.
Used the kifu from high rated games between 2014~2015 on GoQuest.

Note that the kifu after 10 sequences will be used for both of training and test.
