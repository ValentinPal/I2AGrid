# I2AGrid
solving grid environment with imagination augmented agent (I2A).
first step is training an A2C agent. This is done with a2c_train.py which is using the model from models\a2c_model.py and using the configuration from experiment_config.py.
The configuration file has a class for all parameters from all the steps in the same place. it should be split based on the needs.
all files, be it train or test or replay, have command line arguments, some of which are mandatory.

The conv layers definition for all steps the different models used in the 3 different steps are configured in the experminet_config.py

open ai gym env implementation resides in gym-RandomGoalsGrid and should be installed with pip install -e . from the folder that contains the setup.py

experiments that were done were with grid sizes of 5x5, 9x9, 13x13, with resolutions of 14x14, 22x22 and 45x45 respectively. 
Environment returns an observation that includes padding. In a 5x5 grid, the resolution is (5+2)*2.
