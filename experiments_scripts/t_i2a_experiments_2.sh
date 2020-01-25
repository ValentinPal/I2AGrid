ef="runs/Jan24_16-27-37_valy_em_45_13_True/best_5.3155e-05_375797.dat"
g=13
f=45
r=True
python i2a_train.py -s 8000 -r $r -g $g -f $f -e $ef -rs 2
python i2a_train.py -s 8000 -r $r -g $g -f $f -e $ef -rs 2
python i2a_train.py -s 8000 -r $r -g $g -f $f -e $ef -rs 2

python i2a_train.py -s 8000 -r $r -g $g -f $f -e $ef -rs 3
python i2a_train.py -s 8000 -r $r -g $g -f $f -e $ef -rs 3
python i2a_train.py -s 8000 -r $r -g $g -f $f -e $ef -rs 3