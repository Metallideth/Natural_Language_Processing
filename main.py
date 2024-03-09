# Inspired by code at https://colab.research.google.com/github/DhavalTaunk08/Transformers_scripts/blob/master/Transformers_multilabel_distilbert.ipynb#scrollTo=zHxRRzqpBf76

import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
import argparse
import pandas as pd
from datetime import datetime
from netskope_dataloader import NetSkopeDataset
from distilbert_uncased_model import DistilBERTClass
from utils import model_train_loop, model_inference, model_val, impact_eval, antikey_eval, map_historic_to_current_hierarchy, implement_overrides
from model_settings import settings_dict
import pickle
import os
import numpy as np

parser = argparse.ArgumentParser(description='Run model training, including hyperparameter tuning if necessary, as well as testing and inference')
parser.add_argument('-m','--modelmode', help = 'model mode, default = inference_production', default = 'inference_production')
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
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Uncomment above line and comment out below line if desire is to run model on cuda-enabled GPU
DEVICE = torch.device('cpu')
ENCODER = settings_dict['ENCODER']
OVERRIDE_TABLE = settings_dict['OVERRIDE_TABLE']

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
    
if  (MODELMODE == 'inference') or (MODELMODE == 'inference_loss') or (MODELMODE == 'inference_production'):
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
    with open(ENCODER,'rb') as file:
        encoder = pickle.load(file)
    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    # INITIAL DATA FOR INFERENCING STEP IMPORTED IN THE LINE DIRECTLY BELOW
    # INITIAL DATA FOR INFERENCING STEP IMPORTED IN THE LINE DIRECTLY BELOW
    # INITIAL DATA FOR INFERENCING STEP IMPORTED IN THE LINE DIRECTLY BELOW
    # INITIAL DATA FOR INFERENCING STEP IMPORTED IN THE LINE DIRECTLY BELOW
    # INITIAL DATA FOR INFERENCING STEP IMPORTED IN THE LINE DIRECTLY BELOW
    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    # May require updating the NetSkopeDataset module in netskope_dataloader.py to suit your data entry point
    
    data = NetSkopeDataset(INPUTDATA,tokenizer,MAX_LEN)
    
    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    # INITIAL DATA FOR INFERENCING STEP IMPORTED IN THE LINE DIRECTLY ABOVE
    # INITIAL DATA FOR INFERENCING STEP IMPORTED IN THE LINE DIRECTLY ABOVE
    # INITIAL DATA FOR INFERENCING STEP IMPORTED IN THE LINE DIRECTLY ABOVE
    # INITIAL DATA FOR INFERENCING STEP IMPORTED IN THE LINE DIRECTLY ABOVE
    # INITIAL DATA FOR INFERENCING STEP IMPORTED IN THE LINE DIRECTLY ABOVE
    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################

    data_loader = DataLoader(data,**inf_params)
    inf_start = datetime.now().strftime('%d-%m-%Y_%H%M')
    print('Beginning inference...')
    inf_output = model_inference(model=model, data_loader = data_loader,checkpointloc = CHECKPOINTLOC,device=DEVICE,
                                 model_mode = MODELMODE, weights = WEIGHTS,encoder = encoder)
    print('Inference complete.')
    if MODELMODE == 'inference_production':
        for column in inf_output:
            inf_output[column] = inf_output[column].apply(lambda x: encoder[column][x])
        print('Beginning mapping of historic function/role/level hierarchy to go-forward...')
        inf_output = map_historic_to_current_hierarchy(inf_output)
        print('Mapping complete.')
        print('Implementing selected overrides...')
        override_table = pd.read_csv(OVERRIDE_TABLE,encoding='utf-8')
        inf_output = implement_overrides(data.title,inf_output,override_table)
        print('Overrides complete.')
    if 'Unnamed: 0' in inf_output.columns:
        input_with_inf = pd.concat([data.data.drop(columns = 'Unnamed: 0'),inf_output],axis = 1)
    else:
        input_with_inf = pd.concat([data.data,inf_output],axis = 1)
    with open(f'{INFERENCEFOLDER}{inf_start}_inference.pkl','wb') as file:
        pickle.dump(input_with_inf, file)

    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    # FINAL INFERENCE DATA FOR OUTPUT EXPORTED IN THE LINE DIRECTLY BELOW
    # FINAL INFERENCE DATA FOR OUTPUT EXPORTED IN THE LINE DIRECTLY BELOW
    # FINAL INFERENCE DATA FOR OUTPUT EXPORTED IN THE LINE DIRECTLY BELOW
    # FINAL INFERENCE DATA FOR OUTPUT EXPORTED IN THE LINE DIRECTLY BELOW
    # FINAL INFERENCE DATA FOR OUTPUT EXPORTED IN THE LINE DIRECTLY BELOW
    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    # Edit this line to modify export location to suit your needs

    input_with_inf.to_csv(f'{INFERENCEFOLDER}{inf_start}_inference.csv',index=False)
    
    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    # FINAL INFERENCE DATA FOR OUTPUT EXPORTED IN THE LINE DIRECTLY ABOVE
    # FINAL INFERENCE DATA FOR OUTPUT EXPORTED IN THE LINE DIRECTLY ABOVE
    # FINAL INFERENCE DATA FOR OUTPUT EXPORTED IN THE LINE DIRECTLY ABOVE
    # FINAL INFERENCE DATA FOR OUTPUT EXPORTED IN THE LINE DIRECTLY ABOVE
    # FINAL INFERENCE DATA FOR OUTPUT EXPORTED IN THE LINE DIRECTLY ABOVE
    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################

if MODELMODE == 'user_input':
    model = DistilBERTClass()
    model.to(DEVICE)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)
    DIMENSIONS = settings_dict['DIMENSIONS']
    MAX_LEN = settings_dict['MAX_LEN']
    print('Loading model from checkpoint...')
    checkpoint = torch.load(CHECKPOINTLOC, map_location=DEVICE)
    with open(ENCODER,'rb') as file:
        encoder = pickle.load(file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    override_table = pd.read_csv(OVERRIDE_TABLE,encoding='utf-8')
    # Build reverse encoder
    reverse_encoder = {}
    for key in encoder:
        reverse_encoder[key] = {}
        for entry in encoder[key]:
            reverse_encoder[key][encoder[key][entry]] = entry
    # Overwrite when job function = IT and job role = Non-ICP to instead go with the 2nd largest score 
    job_role_nonicp_index = reverse_encoder['Job Role']['NON-ICP']
    job_function_it_index = reverse_encoder['Job Function']['IT']
    job_level_unknown_index = reverse_encoder['Job Level']['UNKNOWN']
    print('Model loaded.')
    print('Enter Job Title and model will output Role, Function, and Level. Press Ctrl + C to quit.')
    try:
        while True:
            job_title = input('Job Title: ')
            job_title = job_title.upper()
            key_outputs = {}
            if job_title in list(override_table.Title):
                this_entry = override_table.loc[override_table.Title == job_title]
                key_outputs['Role'] = this_entry.Role.item()
                key_outputs['Function'] = this_entry.Function.item()
                key_outputs['Level'] = this_entry.Level.item()
            else:
                inputs = tokenizer.encode_plus(job_title,
                                                add_special_tokens = True,
                                                max_length = MAX_LEN,
                                                padding='max_length',
                                                truncation = True)
                ids = torch.tensor(inputs['input_ids']).unsqueeze(0).to(DEVICE)
                mask = torch.tensor(inputs['attention_mask']).unsqueeze(0).to(DEVICE)
                output_logits = model(ids,mask)
                for key in output_logits:
                    #####################################################################################################################
                    #####################################################################################################################
                    #####################################################################################################################
                    #####################################################################################################################
                    # AFTER RETRAINING, MAKE SURE TO MODIFY THE NEXT SECTION
                    # AFTER RETRAINING, MAKE SURE TO MODIFY THE NEXT SECTION
                    # AFTER RETRAINING, MAKE SURE TO MODIFY THE NEXT SECTION
                    # AFTER RETRAINING, MAKE SURE TO MODIFY THE NEXT SECTION
                    # AFTER RETRAINING, MAKE SURE TO MODIFY THE NEXT SECTION
                    #####################################################################################################################
                    #####################################################################################################################
                    #####################################################################################################################
                    #####################################################################################################################
                    # The following code blocks correspond to when running inference production and overwriting results for
                    # Role or Level based on the new hierarchy. When retraining, all of the historical data should be
                    # restated to the new hierarchy anyways, which means that this code should be unecessary once the
                    # model is retrained and a new "final" inference model is created.
                    if key == 'Role':
                        role_top2 = np.array(output_logits[key].argsort(dim=1).detach().cpu())[:,-2:]
                        function = np.array(output_logits['Function'].argmax(dim=1).detach().cpu()).reshape(-1,1)
                        overwrite = (function == job_function_it_index) & (role_top2[:,[-1]] == job_role_nonicp_index)
                        overwrite = overwrite[:,0]
                        # Whenever overwrite is true, we take the 2nd largest value from role. When it's false, we
                        # take the largest. This is akin to passing in 1-overwrite as an index vector for role_top2
                        key_outputs[key] = encoder[f'Job {key}'][role_top2[np.arange(0,role_top2.shape[0]),1-overwrite].item()]
                    elif key == 'Level':
                        level_top2 = np.array(output_logits[key].argsort(dim=1).detach().cpu())[:,-2:]
                        overwrite = level_top2[:,[-1]] == job_level_unknown_index
                        overwrite = overwrite[:,0]
                        # Similar to above rule for role, we overwrite the largest values when overwrite is true
                        key_outputs[key] = encoder[f'Job {key}'][level_top2[np.arange(0,level_top2.shape[0]),1-overwrite].item()]
                    else:
                        key_outputs[key] = encoder[f'Job {key}'][output_logits[key].argmax(dim=1).detach().cpu().item()]
                # Some more custom overwrites
                if key_outputs['Role'] == 'GOVERNANCE RISK COMPLIANCE':
                    key_outputs['Function'] = 'RISK/LEGAL/COMPLIANCE'
                if key_outputs['Function'] != 'IT':
                    key_outputs['Role'] = 'NONE'
                #####################################################################################################################
                #####################################################################################################################
                #####################################################################################################################
                #####################################################################################################################
                # AFTER RETRAINING, MAKE SURE TO MODIFY THE PREVIOUS SECTION
                # AFTER RETRAINING, MAKE SURE TO MODIFY THE PREVIOUS SECTION
                # AFTER RETRAINING, MAKE SURE TO MODIFY THE PREVIOUS SECTION
                # AFTER RETRAINING, MAKE SURE TO MODIFY THE PREVIOUS SECTION
                # AFTER RETRAINING, MAKE SURE TO MODIFY THE PREVIOUS SECTION
                #####################################################################################################################
                #####################################################################################################################
                #####################################################################################################################
                #####################################################################################################################
            print('Predictions:')
            for key in key_outputs:
                print(f'Job {key}: {key_outputs[key]}')
            print('')
    except KeyboardInterrupt:
        pass    
    
if MODELMODE == 'test':
    model = DistilBERTClass()
    model.to(DEVICE)
    MAX_LEN = settings_dict['MAX_LEN']
    TRAIN_BATCH_SIZE = settings_dict['TRAIN_BATCH_SIZE']
    VALID_BATCH_SIZE = settings_dict['VALID_BATCH_SIZE']
    WEIGHTS = settings_dict['WEIGHTS']
    DIMENSIONS = settings_dict['DIMENSIONS']
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)

    test = NetSkopeDataset(INPUTDATA,tokenizer,MAX_LEN)

    test_start = datetime.now().strftime('%d-%m-%Y_%H%M')
    os.mkdir(f'{TESTFOLDER}{test_start}')

    test_params = {
        'batch_size':VALID_BATCH_SIZE,
        'shuffle':False,
        'num_workers':0
    }

    checkpoint = torch.load(CHECKPOINTLOC, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loader = DataLoader(test,**test_params)
    
    test_accuracy, test_loss, test_conf_mat = model_val(0,model,test_loader,WEIGHTS,DIMENSIONS,DEVICE)

    print('Test results - Loss: {:.4f}, Accuracy: Role {:.4f}, Function {:.4f}, Level {:.4f}'.format(test_loss,
        test_accuracy['Role'],test_accuracy['Function'],test_accuracy['Level']))
    test_stats = {
        'test_loss':test_loss,
        'test_accuracy':test_accuracy,
        'test_conf_mat':test_conf_mat
    }
    with open('{}{}/test_summary'.format(TESTFOLDER,test_start),'wb') as file:
            pickle.dump(test_stats, file)

if MODELMODE == 'impact_eval':
    model = DistilBERTClass()
    model.to(DEVICE)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)
    DIMENSIONS = settings_dict['DIMENSIONS']
    MAX_LEN = settings_dict['MAX_LEN']
    IMPACT_EVAL_BATCH_SIZE = settings_dict['IMPACT_EVAL_BATCH_SIZE']
    inf_params = {
        'batch_size':IMPACT_EVAL_BATCH_SIZE,
        'shuffle':False,
        'num_workers':0
    }
    data = NetSkopeDataset(INPUTDATA,tokenizer,MAX_LEN)
    data_loader = DataLoader(data,**inf_params)
    inf_start = datetime.now().strftime('%d-%m-%Y_%H%M')
    with open(ENCODER,'rb') as file:
        encoder = pickle.load(file)
    print('Beginning impact evaluation...')
    inf_output = impact_eval(model=model, data_loader = data_loader,checkpointloc = CHECKPOINTLOC,
                              device=DEVICE,tokenizer = tokenizer,encoder = encoder)
    print('Impact evaluation complete.')
    with open(f'{INFERENCEFOLDER}{inf_start}_impact_output.pkl','wb') as file:
        pickle.dump(inf_output,file)

if MODELMODE == 'antikey_eval':
    model = DistilBERTClass()
    model.to(DEVICE)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)
    DIMENSIONS = settings_dict['DIMENSIONS']
    MAX_LEN = settings_dict['MAX_LEN']
    IMPACT_EVAL_BATCH_SIZE = settings_dict['IMPACT_EVAL_BATCH_SIZE']
    inf_params = {
        'batch_size':IMPACT_EVAL_BATCH_SIZE,
        'shuffle':False,
        'num_workers':0
    }
    data = NetSkopeDataset(INPUTDATA,tokenizer,MAX_LEN)
    data_loader = DataLoader(data,**inf_params)
    inf_start = datetime.now().strftime('%d-%m-%Y_%H%M')
    with open(ENCODER,'rb') as file:
        encoder = pickle.load(file)
    print('Beginning anti-keyword evaluation...')
    inf_output = antikey_eval(model=model, data_loader = data_loader,checkpointloc = CHECKPOINTLOC,
                              device=DEVICE,tokenizer = tokenizer,encoder = encoder)
    print('Anti-keyword evaluation complete.')
    with open(f'{INFERENCEFOLDER}{inf_start}_antikey_output.pkl','wb') as file:
        pickle.dump(inf_output,file)