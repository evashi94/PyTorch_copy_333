from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import visdom
import torch
import torch.nn as nn
import torch.optim as optim

from utils.helpers import loggerConfig

CONFIGS = [
# agent_type, env_type,      game, circuit_type
[ "empty",    "repeat-copy", "",   "none"      ],  # 0 ingore-never use
[ "sl",       "copy",        "",   "ntm"       ],  # 1
[ "sl",       "repeat-copy", "",   "ntm"       ],   # 2 ingore-never use
[ "sl",       "associative-recall", "",   "ntm"       ]  # 3
]

class Params(object):   # NOTE: shared across all modules
    def __init__(self):
        self.verbose     = 0            # 0(warning) | 1(info) | 2(debug)

        # training signature
        self.machine     = "CopyBenchmark"       # "machine_id"
        self.timestamp   = "171202"   # "yymmdd##"

        # training configuration
        self.mode        = 1            # 1(train) | 2(test model_file)
        self.config      = 1            #1:copy #3: associative recall

        self.seed        = 333         #666
        self.render      = True        # whether render the window from the original envs or not
        self.visualize   = True         #whether save bit error on csv  # whether diaplay output and save bit error on command line or not
        self.save_best   = False        # save model w/ highest reward if True, otherwise always save the latest model

        self.agent_type, self.env_type, self.game, self.circuit_type = CONFIGS[self.config]

        self.use_cuda    = torch.cuda.is_available()
        self.dtype       = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        # prefix for model/log/visdom
        self.refs        = self.machine + "_" + self.timestamp  # NOTE: using this as env for visdom
        self.root_dir    = os.getcwd()

        # model files
        # NOTE: will save the current model to model_name
        self.model_name  = self.root_dir + "/models/" + self.refs + ".pth"
        self.model_name_ite  = self.root_dir + "/models/" + self.refs
        # NOTE: will load pretrained model_file if not None
        self.model_file  = None#self.root_dir + "/models/{TODO:FILL_IN_PRETAINED_MODEL_FILE}.pth"
        if self.mode == 2:
            self.model_file  = self.model_name  # NOTE: so only need to change self.mode to 2 to test the current training
            assert self.model_file is not None, "Pre-Trained model is None, Testing aborted!!!"
            self.refs = self.refs + "_test"     # NOTE: using this as env for visdom for testing, to avoid accidentally redraw on the training plots

        # logging configs
        self.log_name    = self.root_dir + "/logs/" + self.refs + ".log"
        self.logger      = loggerConfig(self.log_name, self.verbose)
        self.logger.warning("<===================================>")

        if self.visualize:
            self.vis = visdom.Visdom()
            self.logger.warning("bash$: python -m visdom.server")           # activate visdom server on bash
            self.logger.warning("http://localhost:8097/env/" + self.refs)   # open this address on browser

class EnvParams(Params):    # settings for network architecture
    def __init__(self):
        super(EnvParams, self).__init__()

        self.batch_size = None
        if self.env_type == "copy":
            self.len_word  = 8
            self.min_num_words = 1
            self.max_num_words = 20
            self.save_bit_error = []
            self.save_avg_bit_error = [] 
        elif self.env_type == "associative-recall":
            self.len_word  = 6
            self.min_num_words = 3
            self.max_num_words = 3 #fix the words to be 3
            self.min_num_items = 2
            self.max_num_items = 6
            self.save_bit_error = []
            self.save_avg_bit_error = []
        elif self.env_type == "repeat-copy": #ignore repeat-copy
            self.len_word  = 4
            self.min_num_words = 1
            self.max_num_words = 2
            self.min_repeats   = 1
            self.max_repeats   = 2
            self.max_repeats_norm = 10.

class ControllerParams(Params):
    def __init__(self):
        super(ControllerParams, self).__init__()

        self.batch_size     = None
        self.input_dim      = None  # set after env
        self.read_vec_dim   = None  # num_read_heads x mem_wid
        self.output_dim     = None  # set after env
        self.hidden_dim     = None  #
        self.mem_hei        = None  # set after memory
        self.mem_wid        = None  # set after memory

class HeadParams(Params):
    def __init__(self):
        super(HeadParams, self).__init__()

        self.num_heads = None
        self.batch_size = None
        self.hidden_dim = None
        self.mem_hei = None
        self.mem_wid = None
        self.num_allowed_shifts = 3

class WriteHeadParams(HeadParams):
    def __init__(self):
        super(WriteHeadParams, self).__init__()

class ReadHeadParams(HeadParams):
    def __init__(self):
        super(ReadHeadParams, self).__init__()
        if self.circuit_type == "dnc":
            self.num_read_modes = None

class MemoryParams(Params):
    def __init__(self):
        super(MemoryParams, self).__init__()

        self.batch_size = None
        self.mem_hei = None
        self.mem_wid = None

class AccessorParams(Params):
    def __init__(self):
        super(AccessorParams, self).__init__()

        self.batch_size = None
        self.hidden_dim = None
        self.num_write_heads = None
        self.num_read_heads = None
        self.mem_hei = None
        self.mem_wid = None
        self.clip_value = None
        self.write_head_params = WriteHeadParams()
        self.read_head_params  = ReadHeadParams()
        self.memory_params     = MemoryParams()

class CircuitParams(Params):# settings for network architecture
    def __init__(self):
        super(CircuitParams, self).__init__()

        self.batch_size     = None
        self.input_dim      = None  # set after env
        self.read_vec_dim   = None  # num_read_heads x mem_wid
        self.output_dim     = None  # set after env

        if self.circuit_type == "ntm":
            self.hidden_dim      = 100
            self.num_write_heads = 1
            self.num_read_heads  = 1
            self.mem_hei         = 128
            self.mem_wid         = 20
            self.clip_value      = 20.   # clips controller and circuit output values to in between
        elif self.circuit_type == "dnc":
            self.hidden_dim      = 64
            self.num_write_heads = 1
            self.num_read_heads  = 4
            self.mem_hei         = 16
            self.mem_wid         = 16
            self.clip_value      = 20.   # clips controller and circuit output values to in between

        self.controller_params = ControllerParams()
        self.accessor_params   = AccessorParams()

class AgentParams(Params):  # hyperparameters for drl agents
    def __init__(self):
        super(AgentParams, self).__init__()

        if self.agent_type == "sl":
            if self.circuit_type == "ntm":
                self.criteria       = nn.BCELoss()
                self.optim          = optim.RMSprop

                self.steps          = 300000    # max #iterations
                self.batch_size     = 8         # for stability 16
                self.early_stop     = None      # max #steps per episode
                self.clip_grad      = 50.
                self.lr             = 1e-4
                self.optim_eps      = 1e-10     # NOTE: we use this setting to be equivalent w/ the default settings in tensorflow
                self.optim_alpha    = 0.9       # NOTE: only for rmsprop, alpha is the decay in tensorflow, whose default is 0.9
                self.eval_freq      = 500
                self.eval_steps     = 50
                self.prog_freq      = self.eval_freq
                self.test_nepisodes = 5
            elif self.circuit_type == "dnc":
                self.criteria       = nn.BCELoss()
                self.optim          = optim.RMSprop

                self.steps          = 100000    # max #iterations
                self.batch_size     = 16
                self.early_stop     = None      # max #steps per episode
                self.clip_grad      = 50.
                self.lr             = 1e-4
                self.optim_eps      = 1e-10     # NOTE: we use this setting to be equivalent w/ the default settings in tensorflow
                self.optim_alpha    = 0.9       # NOTE: only for rmsprop, alpha is the decay in tensorflow, whose default is 0.9
                self.eval_freq      = 500
                self.eval_steps     = 50
                self.prog_freq      = self.eval_freq
                self.test_nepisodes = 5
        elif self.agent_type == "empty":
            self.criteria       = nn.BCELoss()
            self.optim          = optim.RMSprop

            self.steps          = 100000    # max #iterations
            self.batch_size     = 16
            self.early_stop     = None      # max #steps per episode
            self.clip_grad      = 50.
            self.lr             = 1e-4
            self.optim_eps      = 1e-10     # NOTE: we use this setting to be equivalent w/ the default settings in tensorflow
            self.optim_alpha    = 0.9       # NOTE: only for rmsprop, alpha is the decay in tensorflow, whose default is 0.9
            self.eval_freq      = 500
            self.eval_steps     = 50
            self.prog_freq      = self.eval_freq
            self.test_nepisodes = 5

        self.env_params     = EnvParams()
        self.circuit_params = CircuitParams()

class Options(Params):
    agent_params  = AgentParams()
