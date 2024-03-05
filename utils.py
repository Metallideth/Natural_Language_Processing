import torch
from tqdm import tqdm
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
import os
from model_settings import settings_dict
import copy

def combined_loss(outputs,targets,weights,reduction='mean'):
    """From the supplied predicted outputs, targets, and weights, computes the weighted cross entropy for each sequence
    
    :param outputs: A dictionary of tensors of logit outputs from a model forward pass
    :type outputs: A dictionary of torch float tensors; each sequence has several values, one for each output class, required
    :param targets: A dictionary of tensors of target indices corresponding to the correct output values
    :type targets: A dictionary of torch int tensors; each sequence has one value for the index of the correct output class, required
    :param weights: A dictionary of weights, one for each of the 3 output fields (Role/Function/Level) to weight for combined loss
    :type weights: A dictionary of floats, required
    :param reduction: A string corresponding to how the cross entropy loss will be reduced, as a full batch will flow into each loss computation. Default is mean.
    :type reduction: A string, optional

    :return: A torch tensor of values corresponding to the combined loss for this batch, if reduced by mean the tensor will have a single value.
    :rtype: A torch tensor of floats
    """
    loss = 0
    for key in targets.keys():
        # Weight cross entropy by category weights
        loss += torch.nn.functional.cross_entropy(outputs[key],targets[key],reduction=reduction)*weights[key]
    return loss

def partial_loss(outputs,targets,reduction='mean'):
    """Computes the loss associated with one of the 3 output branches: Role, Function, or Level, based on what's passed in

    :param outputs: A torch tensor of output logits
    :type outputs: A torch tensor of floats, required
    :param targets: A torch tensor of integer indices corresponding to the correct classification index
    :type targets: A torch tensor of integers, required
    :param reduction: A string corresponding to how the cross entropy loss will be reduced, as a full batch will flow into each loss computation. Default is mean.
    :type reduction: A string, optional

    :return: A torch tensor of values corresponding to the combined loss for this batch, if reduced by mean the tensor will have a single value.
    :rtype: A torch tensor of floats
    """
    # For one single output category
    loss = torch.nn.functional.cross_entropy(outputs,targets,reduction=reduction)
    return loss

def model_inference(model,data_loader,checkpointloc,device,model_mode,weights,encoder):
    """A method to run the various inference modes from the main.py file

    :param model: An initialized model object built from the distilbert_uncased_model.py file
    :type model: DistilBERTClass object, required
    :param data_loader: A dataloader object to feed data into the model
    :type data_loader: torch.utils.data.DataLoader object, required
    :param checkpointloc: The path to the checkpoint to use for inference
    :type checkpointloc: String object, required
    :param device: The device to use for pytorch computations, either cpu or cuda
    :type device: String object, required
    :param model_mode: Model mode configured from parsed arguments from main.py terminal call. See README for details.
    :type model_mode: String object, required
    :param weights: For the inference_loss mode, weights used to determine the combined loss
    :type weights: A dictionary of floats, required
    :param encoder: Dictionary to translate data integer encodings to string labels
    :type encoder: A dictionary of strings mapped to integer keys, required

    :return: A dataframe identical to the one passed in with 3 additional columns corresponding to the Role, Function, and Level predicted. Names will be different depending on model_mode.
    :rtype: Pandas DataFrame.
    """
    if checkpointloc is not None:
        checkpoint = torch.load(checkpointloc)
        model.load_state_dict(checkpoint['model_state_dict'])
    if model_mode == 'inference_production':
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
    model.eval()
    combined_outputs = []
    if model_mode == 'inference_loss':
        combined_loss_outputs = []
    for _,data in tqdm(enumerate(data_loader,0),total=len(data_loader)):
        output_logits = model(data['ids'].to(device),data['mask'].to(device))
        outputs = []
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
            if (model_mode == 'inference_production') & (key == 'Role'):
                role_top2 = np.array(output_logits[key].argsort(dim=1).detach().cpu())[:,-2:]
                function = np.array(output_logits['Function'].argmax(dim=1).detach().cpu()).reshape(-1,1)
                overwrite = (function == job_function_it_index) & (role_top2[:,[-1]] == job_role_nonicp_index)
                overwrite = overwrite[:,0]
                # Whenever overwrite is true, we take the 2nd largest value from role. When it's false, we
                # take the largest. This is akin to passing in 1-overwrite as an index vector for role_top2
                outputs.append(role_top2[np.arange(0,role_top2.shape[0]),1-overwrite].reshape(-1,1))
            elif (model_mode == 'inference_production') & (key == 'Level'):
                level_top2 = np.array(output_logits[key].argsort(dim=1).detach().cpu())[:,-2:]
                overwrite = level_top2[:,[-1]] == job_level_unknown_index
                overwrite = overwrite[:,0]
                # Similar to above rule for role, we overwrite the largest values when overwrite is true
                outputs.append(level_top2[np.arange(0,level_top2.shape[0]),1-overwrite].reshape(-1,1))
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
            else:
                outputs.append(np.array(output_logits[key].argmax(dim=1).detach().cpu()).reshape(-1,1))
        if model_mode == 'inference_loss':
            targets = {k: v.to(device) for k, v in data.items() if k not in ['ids','mask']}
            loss = combined_loss(output_logits,targets, weights, reduction='none')
            combined_loss_outputs.append(np.array(loss.detach().cpu()).reshape(-1,1))
        combined_outputs.append(np.concatenate(outputs,axis=1))
    combined_outputs = pd.DataFrame(np.concatenate(combined_outputs,axis=0))
    if model_mode == 'inference_loss':
        combined_outputs = pd.concat([combined_outputs,pd.DataFrame(np.concatenate(combined_loss_outputs,axis=0))],
                                     axis=1)
    if model_mode == 'inference_production':
        colnames = ["".join(['Job ',key]) for key in output_logits]
    else:
        colnames = ["".join(['Job ',key,' Predicted']) for key in output_logits]
    if model_mode == 'inference_loss':
        colnames.append('Loss')
    combined_outputs.columns = colnames
    return combined_outputs

def model_train(epoch,model,optimizer,train_loader,weights,dimensions,logging,loggingfolder,device,
                train_start,checkpointloc):
    """Method for training through a single epoch

    :param epoch: Number index corresponding to the current epoch
    :type epoch: Integer, required
    :param model: Model to be trained
    :type model: DistilBERTClass object, required
    :param optimizer: Optimizer to manage model updates
    :type optimizer: torch.optim.Adam, required
    :param train_loader: DataLoader object to load in training data
    :type train_loader: torch.utils.data.DataLoader object, required
    :param weights: Initial weights to use for combined cross entropy loss
    :type weights: A dictionary of floats, required
    :param dimensions: Dimensions of output fields
    :type dimensions: A dictionary of integers, required
    :param logging: A boolean flag to turn on/off logging during training
    :type logging: Boolean, required
    :param loggingfolder: Path to the folder to save logging information
    :type loggingfolder: String object, required
    :param device: Device for pytorch computations, either cpu or cuda
    :type device: String object, required
    :param train_start: String detailing the date and time (hours, minutes) the training run started, used to create folder for logging and checkpoints
    :type train_start: String object, required
    :param checkpointloc: Path to checkpoint used as starting point for training
    :type checkpointloc: String object, required

    :return: Updated weights based on last 1000 batches accuracy, average training accuracy across the entire epoch by output key, average combined loss across the entire epoch, and confusion matrices by output key 
    :rtype: A tuple with 4 items: A dictionary of floats, a dictionary of floats, a single float item, and a dictionary of torch integer tensors
    """
    run_accuracy = {}
    avg_accuracy_latest_1000_batches = {}
    run_conf_mat = {}
    conf_mat_latest_1000_batches = {}
    run_loss = []
    avg_loss_latest_1000_batches = []
    for key in weights:
        run_accuracy[key] = []
        avg_accuracy_latest_1000_batches[key] = []
        run_conf_mat[key] = torch.zeros((1,dimensions[key],dimensions[key]))
        conf_mat_latest_1000_batches[key] = torch.zeros((1,dimensions[key],dimensions[key]))
    for _,data in tqdm(enumerate(train_loader, 0)):
        optimizer.zero_grad()
        
        outputs = model(data['ids'].to(device),data['mask'].to(device))
        targets = {k: v.to(device) for k, v in data.items() if k not in ['ids','mask']}

        loss = combined_loss(outputs, targets, weights)
        run_loss.append(loss.item())
        if _ <= 999:
            avg_loss_latest_1000_batches.append(torch.tensor(run_loss).mean().item())
        else:
            avg_loss_latest_1000_batches.append(torch.tensor(run_loss[-1000:]).mean().item())

        for key in targets:
            pred = outputs[key].argmax(axis=1)
            target = targets[key]
            this_accuracy = ((pred == target).sum()/len(target)).item()
            this_dim = dimensions[key]
            run_accuracy[key].append(this_accuracy)
            this_cm = torch.zeros((1,this_dim,this_dim))
            for this_p, this_t in zip(pred,target):
                this_cm[0,this_p.item(),this_t.item()]+=1
            if _ == 0:
                run_conf_mat[key] = this_cm
            else:
                run_conf_mat[key] = torch.cat((run_conf_mat[key],this_cm),dim=0)
            if _ <= 999:
                avg_accuracy_latest_1000_batches[key].append(torch.tensor(run_accuracy[key]).mean().item())
                if conf_mat_latest_1000_batches[key].shape[0] == 1:
                    conf_mat_latest_1000_batches[key]=run_conf_mat[key].sum(dim=0)
                else:
                    conf_mat_latest_1000_batches[key]=torch.cat((conf_mat_latest_1000_batches[key],run_conf_mat[key].sum(dim=0)),dim=0)
            else:
                avg_accuracy_latest_1000_batches[key].append(torch.tensor(run_accuracy[key][-1000:]).mean().item())
                conf_mat_latest_1000_batches[key]=torch.cat((conf_mat_latest_1000_batches[key],run_conf_mat[key][-1000:].sum(dim=0)),dim=0)
        now = datetime.now().strftime('%d-%m-%Y_%H%M')
        if _%1000==0:
            print('Epoch: {:02}, Batch: {:05}, Last 1000 Batches - Loss: {:.4f}, Accuracy: Role ({:.4f}), Function ({:.4f}), Level({:.4f})'.format(epoch,_,avg_loss_latest_1000_batches[-1],avg_accuracy_latest_1000_batches['Role'][-1],avg_accuracy_latest_1000_batches['Function'][-1],avg_accuracy_latest_1000_batches['Level'][-1]))
            if logging:
                log_dict = {}
                for key in weights:
                    log_dict[key] = {
                        'avg_acc_trailing_1000_batches':avg_accuracy_latest_1000_batches[key][-1],
                        'conf_mat_trailing_1000_batches':conf_mat_latest_1000_batches[key][-1],
                    }
                log_dict['avg_loss_trailing_1000_batches'] = avg_loss_latest_1000_batches[-1]
                log_dict['settings'] = settings_dict
                log_dict['checkpoint_start'] = checkpointloc
                with open('{}{}/epoch{:02}_{}_batch{:05}'.format(loggingfolder,train_start,epoch,now,_),'wb') as file:
                    pickle.dump(log_dict, file) 
        else:
            print('Epoch: {:02}, Batch: {:05}, Last 1000 Batches - Loss: {:.4f}, Accuracy: Role ({:.4f}), Function ({:.4f}), Level({:.4f})'.format(epoch,_,avg_loss_latest_1000_batches[-1],avg_accuracy_latest_1000_batches['Role'][-1],avg_accuracy_latest_1000_batches['Function'][-1],avg_accuracy_latest_1000_batches['Level'][-1]),end='\r')


        loss.backward()
        optimizer.step()
        # Update loss function weights to inverse of accuracy - puts more weight on loss associated with what's being
        # incorrectly predicted
        norm_factor = 0
        for key in weights:
            try:
                weights[key] = 1/avg_accuracy_latest_1000_batches[key][-1]
            except ZeroDivisionError:
                weights[key] = 1
            norm_factor += weights[key]
        # Normalize weights so that they add up to 3
        for key in weights:
            weights[key]*=3/norm_factor
    
    epoch_training_accuracy = {}
    epoch_conf_mat = {}
    for key in weights:
        epoch_training_accuracy[key]=torch.tensor(run_accuracy[key]).mean().item()
        epoch_conf_mat[key] = run_conf_mat[key].sum(dim=0)
    epoch_training_loss = torch.tensor(run_loss).mean().item()
    return weights, epoch_training_accuracy, epoch_training_loss, epoch_conf_mat

def model_val(epoch,model,val_loader,weights,dimensions,device):
    """Method for performing validation during training, also used for test

    :param epoch: Number index corresponding to the current epoch
    :type epoch: Integer, required
    :param model: Model to be trained
    :type model: DistilBERTClass object, required
    :param val_loader: DataLoader object to load in validation data
    :type val_loader: torch.utils.data.DataLoader object, required
    :param weights: Initial weights to use for combined cross entropy loss
    :type weights: A dictionary of floats, required
    :param dimensions: Dimensions of output fields
    :type dimensions: A dictionary of integers, required
    :param device: Device for pytorch computations, either cpu or cuda
    :type device: String object, required

    :return: Mean accuracy over the entire dataset, mean loss over the entire dataset, confusion matrix over the entire dataset
    :rtype: Dictionary of individual float items, dictionary of individual float items, dictionary of torch tensor of integers
    """
    run_accuracy = {}
    avg_accuracy_latest_1000_batches = {}
    run_conf_mat = {}
    conf_mat_latest_1000_batches = {}
    run_loss = []
    avg_loss_latest_1000_batches = []
    for key in weights:
        run_accuracy[key] = []
        avg_accuracy_latest_1000_batches[key] = []
        run_conf_mat[key] = torch.zeros((1,dimensions[key],dimensions[key]))
        conf_mat_latest_1000_batches[key] = torch.zeros((1,dimensions[key],dimensions[key]))
    for _,data in tqdm(enumerate(val_loader, 0)):
        
        outputs = model(data['ids'].to(device),data['mask'].to(device))
        targets = {k: v.to(device) for k, v in data.items() if k not in ['ids','mask']}

        loss = combined_loss(outputs, targets, weights)
        run_loss.append(loss.item())
        if _ <= 999:
            avg_loss_latest_1000_batches.append(torch.tensor(run_loss).mean().item())
        else:
            avg_loss_latest_1000_batches.append(torch.tensor(run_loss[-1000:]).mean().item())

        for key in targets:
            pred = outputs[key].argmax(axis=1)
            target = targets[key]
            this_accuracy = ((pred == target).sum()/len(target)).item()
            this_dim = dimensions[key]
            run_accuracy[key].append(this_accuracy)
            this_cm = torch.zeros((1,this_dim,this_dim))
            for this_p, this_t in zip(pred,target):
                this_cm[0,this_p.item(),this_t.item()]+=1
            if _ == 0:
                run_conf_mat[key] = this_cm
            else:
                run_conf_mat[key] = torch.cat((run_conf_mat[key],this_cm),dim=0)
            if _ <= 999:
                avg_accuracy_latest_1000_batches[key].append(torch.tensor(run_accuracy[key]).mean().item())
                if conf_mat_latest_1000_batches[key].shape[0] == 1:
                    conf_mat_latest_1000_batches[key]=run_conf_mat[key].sum(dim=0)
                else:
                    conf_mat_latest_1000_batches[key]=torch.cat((conf_mat_latest_1000_batches[key],run_conf_mat[key].sum(dim=0)),dim=0)
            else:
                avg_accuracy_latest_1000_batches[key].append(torch.tensor(run_accuracy[key][-1000:]).mean().item())
                conf_mat_latest_1000_batches[key]=torch.cat((conf_mat_latest_1000_batches[key],run_conf_mat[key][-1000:].sum(dim=0)),dim=0)
        if _%1000==0:
            print('Epoch: {:02}, Batch: {:05}, Last 1000 Batches - Loss: {:.4f}, Accuracy: Role ({:.4f}), Function ({:.4f}), Level({:.4f})'.format(epoch,_,avg_loss_latest_1000_batches[-1],avg_accuracy_latest_1000_batches['Role'][-1],avg_accuracy_latest_1000_batches['Function'][-1],avg_accuracy_latest_1000_batches['Level'][-1]))
        else:
            print('Epoch: {:02}, Batch: {:05}, Last 1000 Batches - Loss: {:.4f}, Accuracy: Role ({:.4f}), Function ({:.4f}), Level({:.4f})'.format(epoch,_,avg_loss_latest_1000_batches[-1],avg_accuracy_latest_1000_batches['Role'][-1],avg_accuracy_latest_1000_batches['Function'][-1],avg_accuracy_latest_1000_batches['Level'][-1]),end='\r')

    avg_accuracy_overall = {}
    conf_mat_overall = {}
    for key in weights:
        avg_accuracy_overall[key] = torch.tensor(run_accuracy[key]).mean().item()
        conf_mat_overall[key] = run_conf_mat[key].sum(dim=0)
    avg_loss_overall = torch.tensor(run_loss).mean().item()
    return avg_accuracy_overall, avg_loss_overall, conf_mat_overall

def model_train_loop(epochs,model,optimizer,train_loader,val_loader,weights,dimensions,accstop,logging,loggingfolder,checkpointloc,device):
    """Outer loop for training, calls model_train for each epoch and manages per-epoch checkpoints and logging
    
    :param epoch: Number index corresponding to the current epoch
    :type epoch: Integer, required
    :param model: Model to be trained
    :type model: DistilBERTClass object, required
    :param optimizer: Optimizer to manage model updates
    :type optimizer: torch.optim.Adam, required
    :param train_loader: DataLoader object to load in training data
    :type train_loader: torch.utils.data.DataLoader object, required
    :param val_loader: DataLoader object to load in validation data
    :type val_loader: torch.utils.data.DataLoader object, required
    :param weights: Initial weights to use for combined cross entropy loss
    :type weights: A dictionary of floats, required
    :param dimensions: Dimensions of output fields
    :type dimensions: A dictionary of integers, required
    :param accstop: Accuracy for each output key; once all 3 are reached, training stops on that epoch
    :type accstop: A dictionary of integers, required
    :param logging: A boolean flag to turn on/off logging during training
    :type logging: Boolean, required
    :param loggingfolder: Path to the folder to save logging information
    :type loggingfolder: String object, required
    :param checkpointloc: Path to checkpoint used as starting point for training
    :type checkpointloc: String object, required
    :param device: Device for pytorch computations, either cpu or cuda
    :type device: String object, required
    
    :return: None
    """
    epoch_logging_list = []
    train_start = datetime.now().strftime('%d-%m-%Y_%H%M')
    if logging:
        os.mkdir(f'{loggingfolder}{train_start}')
    os.mkdir(f'checkpoints/{train_start}')
    if checkpointloc is not None:
        checkpoint = torch.load(checkpointloc)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
    for epoch in range(epochs):
        model.train()
        print('Training run on epoch {:02}'.format(epoch))
        weights,epoch_training_accuracy,epoch_training_loss,epoch_conf_mat = model_train(epoch,model,optimizer,train_loader,weights,dimensions,logging,loggingfolder,device,train_start,checkpointloc)
            # Re-set loss weights according to accuracy in latest 1000 batches to bring into the validation
        print('Training results - Loss: {:.4f}, Accuracy: Role {:.4f}, Function {:.4f}, Level {:.4f}'.format(epoch_training_loss,
                                                                                                             epoch_training_accuracy['Role'],
                                                                                                            epoch_training_accuracy['Function'],
                                                                                                            epoch_training_accuracy['Level']))
        model.eval()
        print('Validation run on epoch {:02}'.format(epoch))
        val_accuracy, val_loss, val_conf_mat = model_val(epoch,model,val_loader,weights,dimensions,device)
        print('Validation results - Loss: {:.4f}, Accuracy: Role {:.4f}, Function {:.4f}, Level {:.4f}'.format(val_loss,
                                                                                                               val_accuracy['Role'],
                                                                                                               val_accuracy['Function'],
                                                                                                               val_accuracy['Level']))
        epoch_logging_list.append({
            'training_loss':epoch_training_loss,
            'training_accuracy':epoch_training_accuracy,
            'training_conf_mat':epoch_conf_mat,
            'validation_loss':val_loss,
            'validation_accuracy':val_accuracy,
            'validation_conf_mat':val_conf_mat
        })
        if logging:
            with open('{}{}/epoch{:02}_train_val_summary'.format(loggingfolder,train_start,epoch),'wb') as file:
                    pickle.dump(epoch_logging_list, file) 
        torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
            },'checkpoints/{}/epoch{:02}'.format(train_start,epoch))
        acc_flag = True
        for key in weights:
            if val_accuracy[key] < accstop[key]:
                acc_flag = False
        if acc_flag:
            break # accuracy threshold is 

def impact_eval(model,data_loader,checkpointloc,device,tokenizer,encoder):
    """Method for creating the output to be processed into keyword rankings, pairs with notebooks 05 and 10

    :param model: Model to be trained
    :type model: DistilBERTClass object, required
    :param data_loader: DataLoader object to load in input data
    :type data_loader: torch.utils.data.DataLoader object, required
    :param checkpointloc: Path to checkpoint used as starting point for training
    :type checkpointloc: String object, required
    :param device: Device for pytorch computations, either cpu or cuda
    :type device: String object, required
    :param tokenizer: Tokenizer to use for the model
    :type tokenizer: transformers.DistilBertTokenizer object, required
    :param encoder: Dictionary to translate data integer encodings to string labels
    :type encoder: A dictionary of strings mapped to integer keys, required
    
    :return: A list of dictionaries to be unpacked by the associated notebooks
    :rtype: A list of dictionaries
    """
    # Score reduction impact method is my own making
    if checkpointloc is not None:
        checkpoint = torch.load(checkpointloc)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model_augment = copy.deepcopy(model)
    model_augment.eval()
    model_augment.to(device)
    sequence_list = []
    for _,data in tqdm(enumerate(data_loader,0),total=len(data_loader)):
        this_sequence_dict = {}
        logits_dict = {}
        targets = {k: v.to(device) for k, v in data.items() if k not in ['ids','mask']}
        token_ids = data['ids'].squeeze().to(device)
        token_masks_vec = torch.tensor([x not in torch.tensor(tokenizer.all_special_ids).to(device) for x in token_ids]).int().to(device)
        sequence_ex_special_tokens = (token_ids*token_masks_vec)[token_ids*token_masks_vec!=0]
        distinct_tokens = sequence_ex_special_tokens.unique()
        # Decoded tokens
        decoded_sequence = tokenizer.decode(sequence_ex_special_tokens).upper()
        # Model results before zeroing out any word embedding vectors
        output_logits = model(data['ids'].to(device),data['mask'].to(device))
        distinct_tokens_decoded_list = tokenizer.decode(distinct_tokens).upper().split(" ")

        this_sequence_dict['Sequence'] = decoded_sequence

        for decoded_token,token in zip(distinct_tokens_decoded_list,distinct_tokens):
            # Zero out embedded vector corresponding to token in the augmented model
            model_augment.state_dict()['l1.embeddings.word_embeddings.weight'][token,:] = 0.0
            # Retrieve logits with the same sequence, but with the information for that vector zeroed out
            oneout_logits = model_augment(data['ids'].to(device),data['mask'].to(device))
            # Restore augmented model embedded parameters to their prior values
            model_augment.state_dict()['l1.embeddings.word_embeddings.weight'][token,:] = model.state_dict()['l1.embeddings.word_embeddings.weight'][token,:]
            logits_dict[decoded_token] = oneout_logits

        for key in targets:
            # Get the predicted label and the actual label returned here, and a flag if the prediction
            # was correct
            pred_index = output_logits[key].squeeze().argmax().item()
            pred_label = encoder['Job {}'.format(key)][pred_index]
            pred_score = output_logits[key].squeeze().max().item()
            target_label = encoder['Job {}'.format(key)][targets[key].item()]
            correct_prediction = pred_label == target_label
            oneout_scores = []

            # Now determine share of each token in total score reduction when embedded information vector is zeroed
            for decoded_token in distinct_tokens_decoded_list:
                oneout_score = logits_dict[decoded_token][key].squeeze()[pred_index].item()
                oneout_scores.append(oneout_score)
            
            # Now normalize oneout score reduction so that they add up to 1
            # If the score increases, then the score reduction will be negative. This means that the token
            # is working against the predicted answer; I'll zero out the importance of these
            # by applying ReLU
            oneout_score_reduction_raw = pred_score - torch.tensor(oneout_scores).to(device)
            oneout_score_reduction = torch.relu(oneout_score_reduction_raw)
            oneout_scores_norm = oneout_score_reduction/oneout_score_reduction.sum()
            token_ranking = (oneout_scores_norm.argsort(descending=True)+1)

            
            this_sequence_dict[key] = {
                'Prediction':pred_label,
                'Target':target_label,
                'Correct?':correct_prediction,
                'Distinct_Tokens':distinct_tokens_decoded_list,
                'Token_Importance':{k:v.item() for k,v in zip(distinct_tokens_decoded_list,oneout_scores_norm)},
                'Token_Rank':{k:v.item() for k,v in zip(distinct_tokens_decoded_list,token_ranking)},
                'Token_Marginal_Score_Positive':{k:v.item() for k,v in zip(distinct_tokens_decoded_list,oneout_score_reduction)},
                'Token_Marginal_Score_Raw':{k:v.item() for k,v in zip(distinct_tokens_decoded_list,oneout_score_reduction_raw)}
            }
            
        sequence_list.append(this_sequence_dict)

    return sequence_list

def antikey_eval(model,data_loader,checkpointloc,device,tokenizer,encoder):
    """Method for creating the output to be processed into anti-keyword rankings, pairs with notebooks 06 and 11

    :param model: Model to be trained
    :type model: DistilBERTClass object, required
    :param data_loader: DataLoader object to load in input data
    :type data_loader: torch.utils.data.DataLoader object, required
    :param checkpointloc: Path to checkpoint used as starting point for training
    :type checkpointloc: String object, required
    :param device: Device for pytorch computations, either cpu or cuda
    :type device: String object, required
    :param tokenizer: Tokenizer to use for the model
    :type tokenizer: transformers.DistilBertTokenizer object, required
    :param encoder: Dictionary to translate data integer encodings to string labels
    :type encoder: A dictionary of strings mapped to integer keys, required
    
    :return: A list of dictionaries to be unpacked by the associated notebooks
    :rtype: A list of dictionaries
    """
    # Score reduction impact method is my own
    if checkpointloc is not None:
        checkpoint = torch.load(checkpointloc)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model_augment = copy.deepcopy(model)
    model_augment.eval()
    model_augment.to(device)
    sequence_list = []
    for _,data in tqdm(enumerate(data_loader,0),total=len(data_loader)):
        this_sequence_dict = {}
        logits_dict = {}
        targets = {k: v.to(device) for k, v in data.items() if k not in ['ids','mask']}
        token_ids = data['ids'].squeeze().to(device)
        token_masks_vec = torch.tensor([x not in torch.tensor(tokenizer.all_special_ids).to(device) for x in token_ids]).int().to(device)
        sequence_ex_special_tokens = (token_ids*token_masks_vec)[token_ids*token_masks_vec!=0]
        distinct_tokens = sequence_ex_special_tokens.unique()
        # Decoded tokens
        decoded_sequence = tokenizer.decode(sequence_ex_special_tokens).upper()
        # Model results before zeroing out any word embedding vectors
        output_logits = model(data['ids'].to(device),data['mask'].to(device))
        distinct_tokens_decoded_list = tokenizer.decode(distinct_tokens).upper().split(" ")

        this_sequence_dict['Sequence'] = decoded_sequence

        for decoded_token,token in zip(distinct_tokens_decoded_list,distinct_tokens):
            # Zero out embedded vector corresponding to token in the augmented model
            model_augment.state_dict()['l1.embeddings.word_embeddings.weight'][token,:] = 0.0
            # Retrieve logits with the same sequence, but with the information for that vector zeroed out
            oneout_logits = model_augment(data['ids'].to(device),data['mask'].to(device))
            # Restore augmented model embedded parameters to their prior values
            model_augment.state_dict()['l1.embeddings.word_embeddings.weight'][token,:] = model.state_dict()['l1.embeddings.word_embeddings.weight'][token,:]
            logits_dict[decoded_token] = oneout_logits

        for key in targets:
            # Get the predicted label and the actual label returned here, and a flag if the prediction
            # was correct
            max_index = output_logits[key].squeeze().argmax().item()
            # Leave out the max index to see effect on all other
            this_sequence_dict[key] = []
            for pred_index in [x for x in range(len(output_logits[key].squeeze())) if x != max_index]:
                pred_label = encoder['Job {}'.format(key)][pred_index]
                pred_score = output_logits[key].squeeze()[pred_index].item()
                oneout_scores = []

                # Now determine share of each token in total score reduction when embedded information vector is zeroed
                for decoded_token in distinct_tokens_decoded_list:
                    oneout_score = logits_dict[decoded_token][key].squeeze()[pred_index].item()
                    oneout_scores.append(oneout_score)
                
                # This moves the opposite direction of the impact evaluation, we're now looking for large
                # increases in score by taking out the word - that means that the word is decreasing the score
                # a lot, and is thus an anti-keyword
                # Normalize oneout score increase so that they add up to 1
                # If the score decreases, then the score increase will be negative. This means that the token
                # is increasing the prediction of this non-predicted output; I'll zero out the importance of these
                # by applying ReLU, since we're only interested in tokens that are greatly decreasing the probability
                # of non-predicted outputs
                oneout_score_increase_raw = torch.tensor(oneout_scores).to(device) - pred_score
                oneout_score_increase = torch.relu(oneout_score_increase_raw)
                oneout_scores_norm = oneout_score_increase/oneout_score_increase.sum()
                token_ranking = (oneout_scores_norm.argsort(descending=True)+1)

                
                this_sequence_dict[key].append({
                    'Anti-Prediction':pred_label,
                    'Distinct_Tokens':distinct_tokens_decoded_list,
                    'Token_Importance':{k:v.item() for k,v in zip(distinct_tokens_decoded_list,oneout_scores_norm)},
                    'Token_Rank':{k:v.item() for k,v in zip(distinct_tokens_decoded_list,token_ranking)},
                    'Token_Marginal_Score_Positive':{k:v.item() for k,v in zip(distinct_tokens_decoded_list,oneout_score_increase)},
                    'Token_Marginal_Score_Raw':{k:v.item() for k,v in zip(distinct_tokens_decoded_list,oneout_score_increase_raw)}
                })
            
        sequence_list.append(this_sequence_dict)

    return sequence_list

def map_historic_to_current_hierarchy(data):
    """Method to perform some additional steps to convert data outputs from model into the go-forward hierarchy. The next time the model is updated, this method should be removed.

    :param data: Dataframe of output data to change to the new hierarchy
    :type data: Pandas DataFrame, required

    :return: Augmented DataFrame with updated hierarchy
    :rtype: Pandas DataFrame
    """
    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    # NOTE: THIS FUNCTION MAY NO LONGER BE NECESSARY IF MODEL IS RETRAINED ON RENEWED HIERARCHY
    # NOTE: THIS FUNCTION MAY NO LONGER BE NECESSARY IF MODEL IS RETRAINED ON RENEWED HIERARCHY
    # NOTE: THIS FUNCTION MAY NO LONGER BE NECESSARY IF MODEL IS RETRAINED ON RENEWED HIERARCHY
    # NOTE: THIS FUNCTION MAY NO LONGER BE NECESSARY IF MODEL IS RETRAINED ON RENEWED HIERARCHY
    # NOTE: THIS FUNCTION MAY NO LONGER BE NECESSARY IF MODEL IS RETRAINED ON RENEWED HIERARCHY
    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    # Overwrite the function for those that have Role = 'Governance Risk Compliance' to be
    # 'Risk/Legal/Compliance'
    data.loc[data['Job Role'] == 'GOVERNANCE RISK COMPLIANCE','Job Function'] = 'RISK/LEGAL/COMPLIANCE'

    # Overwrite the roles for those that have Function != 'IT' to be 'NONE'
    data.loc[data['Job Function'] != 'IT','Job Role'] = 'NONE'
    return data

def implement_overrides(title,data,override_table):
    """Implements overwriting model results based on entries in overrides table.

    :param title: Column from data table corresponding to input Job Titles, search is run through this to see if it matches any Titles in the overrides table
    :type title: Pandas Series, required
    :param data: Input data with entries to potentially override
    :tyep data: Pandas DataFrame, required
    :param override_table: DataFrame of values to substitute in - these will overwrite entries in data
    :type override_table: Pandas DataFrame, required

    :return: Original data overwritten where deemed necessary by override table
    :rtype: Pandas DataFrame
    """
    for _,row in tqdm(enumerate(override_table.iterrows(),0),total=override_table.shape[0]):
        this_title = row[1].Title
        this_role = row[1].Role
        this_function = row[1].Function
        this_level = row[1].Level
        data.loc[title == this_title,'Job Role'] = this_role
        data.loc[title == this_title,'Job Function'] = this_function
        data.loc[title == this_title,'Job Level'] = this_level
    return data

