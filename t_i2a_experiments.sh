ef="/home/valy/OneDrive/repos/I2A-all/master/runs/Jan19_21-17-43_valy_em_22_9_False/best_1.3618e-07_184447.dat"
g=9
f=22
r=False

python i2a_train.py -s 8000 -r $r -g $g -f $f -e $ef -lr 0.001 -rs 3
python i2a_train.py -s 8000 -r $r -g $g -f $f -e $ef -lr 0.001 -rs 3
python i2a_train.py -s 8000 -r $r -g $g -f $f -e $ef -lr 0.0004 -rs 3
python i2a_train.py -s 8000 -r $r -g $g -f $f -e $ef -lr 0.0004 -rs 3
python i2a_train.py -s 8000 -r $r -g $g -f $f -e $ef -lr 0.0002 -rs 3
python i2a_train.py -s 8000 -r $r -g $g -f $f -e $ef -lr 0.0002 -rs 3
