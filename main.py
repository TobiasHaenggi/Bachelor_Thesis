# Utility
import os, urllib, io
from   datetime import datetime
import numpy as np
import pandas as pd
import gzip
import tarfile
from skimage import io
import warnings

# Pytorch
import torch, torchvision
import torch.nn.functional as F
from   torch import nn, optim
from   torch.autograd import Variable

# Hyperparameters
!pip install ray torch torchvision
import ray
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler

# Evaluation
from   sklearn import metrics
from   sklearn.metrics import classification_report, confusion_matrix

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from   PIL import Image
import visualize
# Custom classes
import Satellite_Data_Set_Class as sds
import ClassNet
import ResNet



##########################
# Reproducability corner #
##########################

seed_value = 42
np.random.seed(seed_value) # set numpy seed
torch.manual_seed(seed_value) # set pytorch seed for the CPU

##############
# Select GPU #
##############

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu').type
torch.cuda.manual_seed(seed_value)

####################
# Data preparation #
####################

path = DIRECTORY_PATH
satellite_csv_file = pd.read_csv(path + 'ben-ge-s_esaworldcover.csv').copy()
satellite_csv_file['patch_id'] = satellite_csv_file['patch_id'].astype(str) + '_all_bands.npy' # necessary as the name in the csv file does not overlap with the actual names of the files

#################
# INSTANTIATION #
#################

raw_dataset = sds(csv_file=satellite_csv_file,
                   root_dir=path)
############
# TRAINING #
############

untrained_model = ResNet.blank_model()
untrained_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(untrained_model.parameters(), lr=0.1)
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
trained_model = ResNet.train_model(untrained_model, criterion, optimizer, trainer, test_quick, step_lr_scheduler, mini_batch_size=256, num_epochs=2)
path = 'model_' + trainer.name + '_' + str(torch_seed) + '.pt'
torch.save(trained_model.state_dict(), path)
