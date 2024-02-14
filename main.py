# Inspired by code at https://colab.research.google.com/github/DhavalTaunk08/Transformers_scripts/blob/master/Transformers_multilabel_distilbert.ipynb#scrollTo=zHxRRzqpBf76

import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from sklearn import metrics
import argparse
# import pickle
from netskope_dataloader import NetSkopeDataset
from distilbert_uncased_model import DistilBERTClass
# from distilbert_uncased_model_frozen import DistilBERTClass
# from distilbert_uncased_model_truncated import DistilBERTClass
from utils import model_train_loop, model_inference
from model_settings import settings_dict

parser = argparse.ArgumentParser(description='Run model training, including hyperparameter tuning if necessary, as well as testing and inference')
parser.add_argument('-r','--randomseed', help = 'Random seed, default = 2024', default = 2024)
parser.add_argument('-m','--modelmode', help = 'model mode, default = training', default = 'training')
# parser.add_argument('-m','--modelmode', help = 'model mode, default = training', default = 'inference')
# parser.add_argument('-ht','--hptune', help = 'if in training mode, tune hyperparameters, or fixed based on user input, default = tune', default = 'tune')
# Probably won't need hyperparameter tuning, model does really well with minimal tuning, and computing resources are a major limitation
parser.add_argument('-l','--logging', help = 'boolean, set to True to compute and save logging outputs, default = True', default = True)
parser.add_argument('-lf','--loggingfolder', help = 'folder path for logging, default = logging/', default = 'logging/')
parser.add_argument('-if','--inffolder', help = 'folder path for saving inference output, default = inference/', default = 'inference/')
parser.add_argument('-tf','--testfolder', help = 'folder path for saving test output, default = test/', default = 'test/')
parser.add_argument('-cl','--checkpointloc', 
                    help = 'location of checkpoint for starting training, or for inference/testing, default = None', 
                    default = None)
# parser.add_argument('-cl','--checkpointloc', 
#                     help = 'location of checkpoint for starting training, or for inference/testing, default = None', 
#                     default = "C:/Users/CoreySarcu/OneDrive - Netskope/netskope/checkpoints/12-02-2024_1415/12-02-2024_2238_epoch00_batch01900")
parser.add_argument('-id','--inputdata', 
                    help = 'path to input data. In case of model mode training, this is the training data. For model mode test, this is the test data. For model mode inference, this is the input data for label prediction, default = Data/train.pkl', 
                    default = 'Data/train.pkl')
parser.add_argument('-vd','--valdata', 
                    help = 'path to validation data, for use in model mode training, default = Data/val.pkl', 
                    default = 'Data/val.pkl')
args = parser.parse_args()
RANDOMSEED = args.randomseed
MODELMODE = args.modelmode
# HPTUNE = args.hptune
LOGGING = args.logging
LOGGINGFOLDER = args.loggingfolder
INFERENCEFOLDER = args.inffolder
TESTFOLDER = args.testfolder
CHECKPOINTLOC = args.checkpointloc
INPUTDATA = args.inputdata
VALDATA = args.valdata
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if MODELMODE == 'training':
    model = DistilBERTClass()
    model.to(DEVICE)
    MAX_LEN = settings_dict['MAX_LEN']
    TRAIN_BATCH_SIZE = settings_dict['TRAIN_BATCH_SIZE']
    VALID_BATCH_SIZE = settings_dict['VALID_BATCH_SIZE']
    EPOCHS = settings_dict['EPOCHS']
    LEARNING_RATE = settings_dict['LEARNING_RATE']
    WEIGHTS = settings_dict['WEIGHTS']
    DIMENSIONS = settings_dict['DIMENSIONS']
    ACCSTOP = settings_dict['ACCSTOP']
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)

    train = NetSkopeDataset(INPUTDATA,tokenizer,MAX_LEN)
    val = NetSkopeDataset(VALDATA,tokenizer,MAX_LEN)

    train_params = {
        'batch_size':TRAIN_BATCH_SIZE,
        'shuffle':True,
        'num_workers':0
    }
    val_params = {
        'batch_size':VALID_BATCH_SIZE,
        'shuffle':True,
        'num_workers':0
    }

    train_loader = DataLoader(train,**train_params)
    val_loader = DataLoader(val,**val_params)

    optimizer = torch.optim.Adam(params = model.parameters(),lr=LEARNING_RATE)

    model_train_loop(epochs=EPOCHS, model=model, optimizer = optimizer, train_loader = train_loader, weights = WEIGHTS,
                     dimensions = DIMENSIONS,accstop=ACCSTOP,logging=LOGGING,loggingfolder=LOGGINGFOLDER,
                     checkpointloc = CHECKPOINTLOC)
    
if  MODELMODE == 'inference':
    model = DistilBERTClass()
    model.to(DEVICE)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)
    WEIGHTS = settings_dict['WEIGHTS']
    DIMENSIONS = settings_dict['DIMENSIONS']
    MAX_LEN = settings_dict['MAX_LEN']

    data = NetSkopeDataset(INPUTDATA,tokenizer,MAX_LEN)
    
    model_inference(model=model, data = data, weights = WEIGHTS,dimensions = DIMENSIONS,checkpointloc = CHECKPOINTLOC)
    

