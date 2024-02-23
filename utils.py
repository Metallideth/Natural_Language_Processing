import torch
from tqdm import tqdm
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
import os
from model_settings import settings_dict
from collections import defaultdict

def combined_loss(outputs,targets,weights,reduction='mean'):
    loss = 0
    for key in targets.keys():
        loss += torch.nn.functional.cross_entropy(outputs[key],targets[key],reduction=reduction)*weights[key]
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

def gradcam_eval(model,data_loader,checkpointloc,device,weights,tokenizer):
    # inspired by code at https://medium.com/apache-mxnet/let-sentiment-classification-model-speak-for-itself-using-grad-cam-88292b8e4186
    if checkpointloc is not None:
        checkpoint = torch.load(checkpointloc)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    sequence_list = []
    for _,data in tqdm(enumerate(data_loader,0),total=len(data_loader)):
        output_logits = model(data['ids'].to(device),data['mask'].to(device))
        targets = {k: v.to(device) for k, v in data.items() if k not in ['ids','mask']}
        loss = combined_loss(output_logits,targets, weights, reduction='none')
        loss.backward()
        # The layer we're interested in is the word embeddings layer
        mid_layer = model.l1.embeddings.word_embeddings
        # Calculate the gradient for each embedding based on the loss. Only the gradients corresponding to
        # tokens that were included in the input will be nonzero
        token_grad_mean = mid_layer.weight.grad.mean(axis = 1).detach()
        # Normalizing by number of tokens, including special [CLS] (start) and [SEP] (end) tokens
        global_grad_mean = token_grad_mean/(data['mask'].sum())
        # Since the above is a gradient vector equal to the length of the embedded vocabulary, use data['ids'] to
        # pull the correct indices
        token_grads = global_grad_mean[data['ids']].unsqueeze(2)
        activations = model.l1.embeddings.word_embeddings(data['ids'].to(device)).detach()
        # Broadcast both together and sum up by token
        token_grad_act_product = (token_grads * activations).sum(axis = 2)
        # Take relu - only positive results returned
        positive_token_product = torch.relu(token_grad_act_product)
        # Normalize results to add up to 100
        normalized_token_scores = (positive_token_product/positive_token_product.sum()).squeeze()
        # Return decoded tokens for this sequence
        decoded_tokens = tokenizer.decode(data['ids'].squeeze()).upper()
        decoded_tokens_list = decoded_tokens.split(" ")
        # Create a dictionary of nonzero tokens and their associated values
        # This essentially multiplies the effect of words that appear multiple times in the sequence
        # This isn't perfect, but I'm going to assume this doesn't happen that much
        this_sequence_dict = defaultdict(int)
        for token,importance in zip(decoded_tokens_list,normalized_token_scores.cpu().numpy()):
            if importance != 0:
                this_sequence_dict[token] += importance 
        sequence_list.append(this_sequence_dict)

    return sequence_list