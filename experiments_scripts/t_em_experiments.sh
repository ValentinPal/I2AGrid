g=13
f=45
python em_train.py -r True -g $g -f $f -s 200000 -a "runs/Jan24_01-55-12_valy_a2c_45_13_True/best_0001.000_1750.dat"
python em_train.py -r True -g $g -f $f -s 200000 -a "runs/Jan24_01-55-12_valy_a2c_45_13_True/best_0002.000_2250.dat"
python em_train.py -r True -g $g -f $f -s 200000 -a "runs/Jan24_01-55-12_valy_a2c_45_13_True/best_0003.400_3250.dat"
