import torch
import sys
import time, datetime
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve
from torch.autograd import Variable
import numpy as np
import gc
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline



import logging
sys.path.insert(0, './utils')
import config_parser
import PawsDataSet
sys.path.insert(1, './models')
import Bert

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6"
gpu_ids = [0,1,2,3,4,5]

CUDA = False
if torch.cuda.is_available():
    CUDA = True

def _train(config):
    
    model = Bert.load_model(config.reload_model_name, CUDA, gpu_ids)

    Bert.train(PawsDataSet, model, config)



def _test(config):
    
   return  


if __name__ == '__main__':
  config = config_parser.parser.parse_args()
  logging.basicConfig(
    filename= config.log_dir + "/"+ config.model_id + datetime.datetime.now().strftime("_%m-%d_%H") + config.mode + '.log',
    level=logging.INFO, format='%(asctime)s %(name)-8s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M')
  logging.info(config)
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  torch.cuda.manual_seed(config.seed)

  if config.mode == 'train':
    _train(config)
  elif config.mode == 'test':
    _test(config)
  else:
    raise ValueError("invalid value for 'mode': {}".format(config.mode))
