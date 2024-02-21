# Inspired by code at https://colab.research.google.com/github/DhavalTaunk08/Transformers_scripts/blob/master/Transformers_multilabel_distilbert.ipynb#scrollTo=zHxRRzqpBf76

import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from sklearn import metrics
import argparse
import pandas as pd
from datetime import datetime
# import pickle
from netskope_dataloader import NetSkopeDataset
from distilbert_uncased_model import DistilBERTClass
# from distilbert_uncased_model_frozen import DistilBERTClass
# from distilbert_uncased_model_truncated import DistilBERTClass
from utils import model_train_loop, model_inference
from model_settings import settings_dict
import pickle

parser = argparse.ArgumentParser(description='Run model training, including hyperparameter tuning if necessary, as well as testing and inference')
parser.add_argument('-m','--modelmode', help = 'model mode, default = training', default = 'training')
# parser.add_argument('-m','--modelmode', help = 'model mode, default = training', default = 'user_input')
parser.add_argument('-l','--logging', help = 'boolean, set to True to compute and save logging outputs, default = True', default = True)
parser.add_argument('-id','--inputdata', 
                    help = 'path to input data. In case of model mode training, this is the training data. For model mode test, this is the test data. For model mode inference, this is the input data for label prediction, default = Data/train.pkl', 
                    default = 'Data/train.pkl')
parser.add_argument('-vd','--valdata', 
                    help = 'path to validation data, for use in model mode training, default = Data/val.pkl', 
                    default = 'Data/val.pkl')
args = parser.parse_args()
RANDOMSEED = settings_dict['RANDOMSEED']
MODELMODE = args.modelmode
LOGGING = args.logging
LOGGINGFOLDER = settings_dict['LOGGINGFOLDER']
INFERENCEFOLDER = settings_dict['INFERENCEFOLDER']
TESTFOLDER = settings_dict['TESTFOLDER']
CHECKPOINTLOC = settings_dict['CHECKPOINTLOC']
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

    model_train_loop(epochs=EPOCHS, model=model, optimizer = optimizer, train_loader = train_loader, val_loader = val_loader,
                    weights = WEIGHTS,
                     dimensions = DIMENSIONS,accstop=ACCSTOP,logging=LOGGING,loggingfolder=LOGGINGFOLDER,
                     checkpointloc = CHECKPOINTLOC, device = DEVICE)
    
if  (MODELMODE == 'inference') or (MODELMODE == 'inference_loss'):
    model = DistilBERTClass()
    model.to(DEVICE)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)
    WEIGHTS = settings_dict['INF_WEIGHTS']
    DIMENSIONS = settings_dict['DIMENSIONS']
    MAX_LEN = settings_dict['MAX_LEN']
    INF_BATCH_SIZE = settings_dict['INF_BATCH_SIZE']
    inf_params = {
        'batch_size':INF_BATCH_SIZE,
        'shuffle':False,
        'num_workers':0
    }
    data = NetSkopeDataset(INPUTDATA,tokenizer,MAX_LEN)
    data_loader = DataLoader(data,**inf_params)
    inf_start = datetime.now().strftime('%d-%m-%Y_%H%M')
    print('Beginning inference...')
    inf_output = model_inference(model=model, data_loader = data_loader,checkpointloc = CHECKPOINTLOC,device=DEVICE,
                                 model_mode = MODELMODE, weights = WEIGHTS)
    print('Inference complete.')
    input_with_inf = pd.concat([data.data,inf_output],axis = 1)
    with open(f'{INFERENCEFOLDER}{inf_start}_inference.pkl','wb') as file:
        pickle.dump(input_with_inf, file)
    input_with_inf.to_csv(f'{INFERENCEFOLDER}{inf_start}_inference.csv')

if MODELMODE == 'user_input':
    model = DistilBERTClass()
    model.to(DEVICE)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)
    DIMENSIONS = settings_dict['DIMENSIONS']
    MAX_LEN = settings_dict['MAX_LEN']
    print('Loading model from checkpoint...')
    checkpoint = torch.load(CHECKPOINTLOC)
    with open('./Data/index_label_mapping.pkl','rb') as file:
        encoder = pickle.load(file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print('Model loaded.')
    print('Enter Job Title and model with output Role, Function, and Level. Press Ctrl + C to quit.')
    try:
        while True:
            job_title = input('Job Title: ')
            inputs = tokenizer.encode_plus(job_title,
                                            add_special_tokens = True,
                                            max_length = MAX_LEN,
                                            padding='max_length',
                                            truncation = True)
            ids = torch.tensor(inputs['input_ids']).unsqueeze(0).to(DEVICE)
            mask = torch.tensor(inputs['attention_mask']).unsqueeze(0).to(DEVICE)
            output_logits = model(ids,mask)
            print('Predictions:')
            for key in output_logits:
                pred = encoder[f'Job {key}'][output_logits[key].argmax(dim=1).item()]
                print(f'Job {key}: {pred}')
            print('')
    except KeyboardInterrupt:
        pass    
    

