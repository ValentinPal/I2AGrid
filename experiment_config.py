import random

class ExperimentCfg():
    def __init__(self):
        self.GAMMA=0.99 #gamma discount for computing the discounted rewards
        self.GRID_SIZE = 5
        self.A2C_FILE_NAME = "" #file name of the a2c model needed for the EM training
        self.EM_FILE_NAME = "" #file name of the env model needed for i2a training
        self.LEARNING_RATE=0.0008 #learning rate
        self.NUM_ENVS=32 #number of parallel environments to sample for experience tuples
        self.REWARD_STEPS=3 #number of real steps in the simulated trajectory - n-step td
        self.BATCH_SIZE=64
        self.TEST_EVERY_BATCH=250 # after how many batches to perform test on the model
        self.SEED=20 #default seed to perform the experiment with
        self.ENTROPY_BETA=0.015 #weight of the entropy when incorporating it into the loss
        self.VALUE_LOSS_COEF=0.5 #weight of the value state when computing the loss
        self.ENV_NAME='RandomGoalsGrid3CFast-v0'
        self.CLIP_GRAD=1 #how much to clip the gradient when doing backprop and updating the weights
        self.CHANNELS=3 #number of channels of the image
        self.FRAMES_COUNT=1 #number of frames to stack for a single fw pass, to let the net take advantage of the dynamics of the env
        self.FRAME_SIZE=14 #env image res
        self.PARTIALLY_OBSERVED_GRID=False
        self.USE_FRAMESTACK_WRAPPER=True
        self.CONV_LARGE=False #use large conv model
        self.PIXELS_ENV=True #wether to use regular image or 1 channel matrix with 1 input per grid object
        self.REPLACEMENT=True # if to replace the eaten square or not
        self.NEGATIVE_RW=-1
        self.POSITIVE_RW=1
        self.A2C_CONV_LAYERS= [
                {'in_c': 3, 'out_c': 64, 'k': 5, 's': 2, 'p':0},
                {'in_c': 64, 'out_c': 64, 'k': 3, 's': 2, 'p':0},
                {'in_c': 64, 'out_c': 64, 'k': 3, 's': 1, 'p': 0}
                # {'in_c': 64, 'out_c': 64, 'k': 3, 's': 1, 'p':1},
                # {'in_c': 64, 'out_c': 64, 'k': 3, 's': 1}
        ]
        self.EM_CONV1 = [
                {'in_c': 3, 'out_c': 64, 'k': 4, 's': 2 , 'p':0},
                {'in_c': 64 , 'out_c': 64, 'k': 3, 's': 1, 'p':1},
                {'in_c': 64, 'out_c': 64, 'k': 3, 's': 1, 'p': 1}
        ]
        self.EM_CONV2 = {'in_c': 64, 'out_c': 64,'k': 3, 'p':1}
        self.EM_DECONV = {'in_c': 64, 'out_c': 3, 'k': 4, 's': 2, 'p': 0}
        self.EM_RW_FC = 512
        self.FC_LAYER = 256
        self.POLICY_LAYER=256
        self.VALUE_LAYER= 256
        self.HOOK_SWITCH=False
        self.OBS_WEIGHT=10.0 #weight of the observation prediction in the loss calculation in the EM
        self.REWARD_WEIGHT=1.0
        self.SAVE_EVERY_BATCH=1000 #how often to save the model
        self.ROLL_STEPS=1
        self.POLICY_LR=1e-4 #LR for the policy distillation optimization
        self.DEVICE='cuda'
        self.IMG_SHAPE = (self.CHANNELS * self.FRAMES_COUNT, self.FRAME_SIZE, self.FRAME_SIZE)
        self.A2CNET = ""
        self.EM_NET = ""
        self.I2A_NET = ""
        self.ROLLOUT_HIDDEN = 512

    def make_base_config(self, args):
        SEED = None
        if(args.FORCE_SEED):
            SEED = args.SEED
        else:
            SEED = random.randint(0, 2**32 - 1)

        self.SEED = SEED
        self.GRID_SIZE = args.GRID_SIZE
        self.CUDA = args.CUDA
        self.FRAME_SIZE = args.FRAME_SIZE
        self.IMG_SHAPE = (self.CHANNELS * self.FRAMES_COUNT, self.FRAME_SIZE, self.FRAME_SIZE)
        self.REPLACEMENT = self.str_to_bool(args.REPLACEMENT)
        print(args.REPLACEMENT, type(args.REPLACEMENT))
        print(self.REPLACEMENT, type(self.REPLACEMENT))
        assert (isinstance(self.REPLACEMENT, bool))

    def str_to_bool(self, arg):
        if(arg == "True"):
            return True
        else:
            return False

    def make_a2c_config(self, parser):
        args = parser.parse_args()
        self.make_base_config(args)
        self.STEPS = args.STEPS


    def make_em_config(self, parser):
        args = parser.parse_args()
        self.make_base_config(args)
        self.A2C_FILE_NAME = args.A2C_FILE_NAME
        self.EM_STEPS = args.EM_STEPS


    def make_i2a_config(self, parser):
        args = parser.parse_args()
        self.make_base_config(args)
        self.EM_FILE_NAME = args.EM_FILE
        self.ROLL_STEPS = args.ROLL_STEPS
        self.STEPS = args.STEPS
        self.ROLLOUT_ENCODER = ""

    def make_replay_config(self, parser):
        args = parser.parse_args()
        self.make_base_config(args)
        self.EM_FN= args.EM_FILE
        self.A2C_FN = args.A2C_FILE
        self.I2A_FN = args.I2A_FILE
        self.EPISODES = args.EPISODES
        self.PLOT=self.str_to_bool(args.PLOT)
        self.INPUT=self.str_to_bool(args.INPUT)

    def make_test_env_config(self, parser):
        args = parser.parse_args()
        self.make_base_config(args)
        self.INPUT = args.INPUT
        self.EPISODES = args.EPISODES
        self.A2C_FN = args.A2C_FILE
        self.PLOT=self.str_to_bool(args.PLOT)
        self.INPUT = self.str_to_bool(args.INPUT)

    def build_name_for_writer(self):
        name = str(self.FRAME_SIZE) + "_" \
               + str(self.GRID_SIZE) + "_" + str(self.REPLACEMENT)

        return name

    def build_name_for_i2a_writer(self):
        name = str(self.FRAME_SIZE) + "_" \
               + str(self.GRID_SIZE) + "_" \
               + str(self.ROLL_STEPS) + "_"\
                + str(self.REPLACEMENT)

        return name

