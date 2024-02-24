import torch
from tqdm import tqdm
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
import os
from model_settings import settings_dict
from collections import defaultdict
import copy

def combined_loss(outputs,targets,weights,reduction='mean'):
    loss = 0
    for key in targets.keys():
        loss += torch.nn.functional.cross_entropy(outputs[key],targets[key],reduction=reduction)*weights[key]
    return loss

def partial_loss(outputs,targets,reduction='mean'):
    loss = torch.nn.functional.cross_entropy(outputs,targets,reduction=reduction)
    return loss

def model_inference(model,data_loader,checkpointloc,device,model_mode,weights):
    
    if checkpointloc is not None:
        checkpoint = torch.load(checkpointloc)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    combined_outputs = []
    if model_mode == 'inference_loss':
        combined_loss_outputs = []
    for _,data in tqdm(enumerate(data_loader,0)):
        output_logits = model(data['ids'].to(device),data['mask'].to(device))
        outputs = []
        for key in output_logits:
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
    colnames = ["".join(['Job ',key,' Predicted']) for key in output_logits]
    if model_mode == 'inference_loss':
        colnames.append('Loss')
    combined_outputs.columns = colnames
    return combined_outputs

def model_train(epoch,model,optimizer,train_loader,weights,dimensions,logging,loggingfolder,device,
                train_start,checkpointloc):
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
            break # accuracy threshold is reached

def gradcam_eval(model,data_loader,checkpointloc,device,weights,tokenizer,encoder):
    # inspired by code at https://medium.com/apache-mxnet/let-sentiment-classification-model-speak-for-itself-using-grad-cam-88292b8e4186
    # GradCAM explained in https://arxiv.org/pdf/1610.02391.pdf for image modeling
    if checkpointloc is not None:
        checkpoint = torch.load(checkpointloc)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model_augment = copy.deepcopy(model)
    model_augment.eval()
    sequence_list = []
    for _,data in tqdm(enumerate(data_loader,0),total=len(data_loader)):
        this_sequence_dict = {}
        targets = {k: v.to(device) for k, v in data.items() if k not in ['ids','mask']}
        for key in targets:
            # Needed to avoid infinite backpropagation graph, we recreate outputs every time
            # this loop runs
            output_logits = model(data['ids'].to(device),data['mask'].to(device))
            # Get the predicted label and the actual label returned here, and a flag if the prediction
            # was correct
            pred_label = encoder['Job {}'.format(key)][output_logits[key].squeeze().argmax().item()]
            target_label = encoder['Job {}'.format(key)][targets[key].item()]
            correct_prediction = pred_label == target_label
            # Pull the non-normalized logit for the prediction to form the endpoint to backpropagate
            # from
            output_pred_score = output_logits[key].squeeze().max()
            # Create a dictionary of nonzero tokens and their associated values
            # This essentially multiplies the effect of words that appear multiple times in the sequence
            # This isn't perfect, but I'm going to assume this doesn't happen that much
            output_pred_score.backward()
            # The layer we're interested in is the word embeddings layer
            token_ids = data['ids'].squeeze().to(device)
            distinct_tokens = token_ids[token_masks_vec.bool()].unique()
            # For factor importance, we will use the method of integrated gradients, explained here:
            # https://medium.com/@madhubabu.adiki/integrated-gradients-for-natural-language-processing-from-scratch-c81c50c5bc4d
            word_embeddings = model.l1.embeddings.word_embeddings
            # We'll also add position embeddings - this contains information of where the word appears in the sequence
            # The embedded representation of each token is a sum of word and position embeddings
            # For tokens that appear multiple times in a sequence, the only difference is the position embedding. 
            # In consideration of this, I'll average the embeddings for that token.
            position_embeddings = model.l1.embeddings.position_embeddings
            # The word embeddings contains an entry for every single token in the entire vocabulary, so we only want to
            # pull the embeddings that appear in this sequence. We also only want to pull the position embeddings for
            # the first 64 tokens, since that's the max sequence length we have
            # For my purposes, I will zero out everything except for the tokens corresponding to input words
            token_masks_vec = torch.tensor([x not in torch.tensor(tokenizer.all_special_ids).to(device) for x in token_ids]).int().to(device)
            token_masks = token_masks_vec.unsqueeze(1)
            token_embeddings = (word_embeddings.weight[token_ids] + position_embeddings.weight[:len(token_ids),:]).detach()*token_masks
            # There is also a sequence embedding for samples with multiple sequences, but this is not an issue here since
            # there's only one sequence per sample in our data, and Distilbert also removed this aspect from the larger
            # BERT model anyways.
            # Calculate the gradient for each embedding based on the output predicted score
            token_grad = (word_embeddings.weight.grad[token_ids] + position_embeddings.weight.grad[:len(token_ids),:]).detach()*token_masks
            # Before continuing, we will consolidate any instances of tokens that appear multiple times in the text by averaging them
            token_embeddings_condensed = torch.tensor([]).to(device)
            token_grad_condensed = torch.tensor([]).to(device)
            for token in distinct_tokens:
                this_index = token_ids == token
                this_token_embeddings = token_embeddings[this_index]
                this_token_grad = token_grad[this_index]
                token_embeddings_condensed = torch.concat((token_embeddings_condensed,
                                                           this_token_embeddings.mean(axis=0).unsqueeze(0)))
                token_grad_condensed = torch.concat((token_grad_condensed,
                                                     this_token_grad.mean(axis=0).unsqueeze(0)))
            # This is where I diverge from what GradCAM is doing. Ultimately, what I'm interested in is a token that
            # has very high impact on the positive predictiveness of the predicted result. In other words, a token that
            # has very high gradients relative to the output prediction. Consider that embeddings are appropriate and
            # shouldn't shift much in any direction; a sharp gradient would indicate that movement in that particular
            # weight would result in a relatively large movement in the output. Assuming the output is correct,
            # positive gradients signify that the weight should increase and negative gradients indicate that the weight should
            # decrease. Embedded weights that are all zero effectively neutralize the impact of an input token. Therefore, we
            # need to find some way to indicate either (1) high absolute value of embeddings with little to no gradien
            token_avg_importance = (token_embeddings_condensed*token_grad_condensed).mean(axis=1)
            # Normalize by the sum to get token importance
            token_importance_norm = token_avg_importance.abs()/token_avg_importance.abs().sum()
            # Decoded tokens
            decoded_tokens = tokenizer.decode(distinct_tokens).upper().split(" ")
            decoded_sequence = tokenizer.decode((token_ids*token_masks_vec)[token_ids*token_masks_vec!=0]).upper().split(" ")
            this_sequence_dict[key] = {
                'Sequence':decoded_sequence,
                'Prediction':pred_label,
                'Target':target_label,
                'Correct?':correct_prediction,
                'Distinct_Tokens':decoded_tokens,
                'Importance':{k:v.item() for k,v in zip(decoded_tokens,token_importance_norm)}
            }
            
        sequence_list.append(this_sequence_dict)

    return sequence_list

