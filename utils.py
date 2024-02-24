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
from transformers import DistilBertModel
import captum
from captum.attr import IntegratedGradients
import distilbert_uncased_model_role, distilbert_uncased_model_function, distilbert_uncased_model_level

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

def gradcam_eval(model,data_loader,checkpointloc,device,tokenizer,encoder):
    # inspired by code at https://medium.com/apache-mxnet/let-sentiment-classification-model-speak-for-itself-using-grad-cam-88292b8e4186
    # GradCAM explained in https://arxiv.org/pdf/1610.02391.pdf for image modeling
    if checkpointloc is not None:
        checkpoint = torch.load(checkpointloc)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model_augment = copy.deepcopy(model)
    model_augment.eval()
    base_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    base_model.to(device)
    # Modify state dict to change parameters
    # model_augment.state_dict()['l1.embeddings.word_embeddings.weight'][0,0]=50000000
    sequence_list = []
    model_role = distilbert_uncased_model_role.DistilBERTClass()
    model_function = distilbert_uncased_model_function.DistilBERTClass()
    model_level = distilbert_uncased_model_level.DistilBERTClass()
    model_role.to(device)
    model_function.to(device)
    model_level.to(device)
    # model_role.children() = [x for x in model.children]
    role_state_dict = model_role.state_dict()
    function_state_dict = model_function.state_dict()
    level_state_dict = model_level.state_dict()
    for role_key, function_key, level_key in zip(model_role.state_dict(), model_function.state_dict(), model_level.state_dict()):
        role_state_dict[role_key] = model.state_dict()[role_key]
        function_state_dict[function_key] = model.state_dict()[function_key]
        level_state_dict[level_key] = model.state_dict()[level_key]
    model_role.load_state_dict(role_state_dict)
    model_function.load_state_dict(function_state_dict)
    model_level.load_state_dict(level_state_dict)
    model_role.eval()
    model_function.eval()
    model_level.eval()
    ig = IntegratedGradients(model_role)
    for _,data in tqdm(enumerate(data_loader,0),total=len(data_loader)):
        this_sequence_dict = {}
        targets = {k: v.to(device) for k, v in data.items() if k not in ['ids','mask']}
        for key in targets:
            # Needed to avoid infinite backpropagation graph, we recreate outputs every time
            # this loop runs
            output_logits = model(data['ids'].to(device),data['mask'].to(device))
            # Get the predicted label and the actual label returned here, and a flag if the prediction
            # was correct
            pred_index = output_logits[key].squeeze().argmax().item()
            pred_label = encoder['Job {}'.format(key)][pred_index]
            pred_score = output_logits[key].squeeze().max()
            target_label = encoder['Job {}'.format(key)][targets[key].item()]
            correct_prediction = pred_label == target_label
            token_ids = data['ids'].squeeze().to(device)
            token_masks_vec = torch.tensor([x not in torch.tensor(tokenizer.all_special_ids).to(device) for x in token_ids]).int().to(device)
            # Decoded tokens
            sequence_ex_special_tokens = (token_ids*token_masks_vec)[token_ids*token_masks_vec!=0]
            distinct_tokens = sequence_ex_special_tokens.unique()
            decoded_sequence = tokenizer.decode(sequence_ex_special_tokens).upper()
            # Now to calculate integrated gradients as described in this article
            # https://arxiv.org/pdf/2108.13654.pdf
            weight_tensor = torch.arange(51)/50 #50-step integrated gradient
            end_embeddings = model.state_dict()['l1.embeddings.word_embeddings.weight'][distinct_tokens]
            start_embeddings = base_model.state_dict()['embeddings.word_embeddings.weight'][distinct_tokens]
            integrated_gradients = torch.zeros(end_embeddings.shape).to(device)
            # Calculate integrated gradients, one token at a time
            for index,token in enumerate(distinct_tokens):
                # Set embedding vector to baseline vector
                model_augment.state_dict()['l1.embeddings.word_embeddings.weight'][token] = base_model.state_dict()[
                    'embeddings.word_embeddings.weight'][token]
                for weight in weight_tensor:
                    # Since baseline is zero vector in this case, we can re-define word embeddings as weight multiplied
                    # by their current embedding
                    model_augment.state_dict()['l1.embeddings.word_embeddings.weight'][token] += weight*(
                        end_embeddings[index,:]-start_embeddings[index,:])
                    # Run forward pass
                    output_logits_ig = model_augment(data['ids'].to(device),data['mask'].to(device))
                    # Retrieve predicted label score
                    pred_label_score = output_logits_ig[key].squeeze()[pred_index]
                    # Run backwards pass
                    pred_label_score.backward()
                    # Retrieve gradient
                    current_embeddings_grad = model_augment.l1.embeddings.word_embeddings.weight.grad[token]
                    # Integrate the gradient
                    integrated_gradients[index,:]+=(current_embeddings_grad*((
                        end_embeddings[index,:]-start_embeddings[index,:])/len(weight_tensor)))
                model_augment.state_dict()['l1.embeddings.word_embeddings.weight'][token] = end_embeddings[index,:]

            # Since baseline is zero, we can complete integrated gradients by multiplying by end embeddings
                
            gradient_token_index = torch.tensor([x in sequence_ex_special_tokens for x in distinct_tokens]).to(device)
            unnormalized_token_scores = integrated_gradients.mean(axis = 1)[gradient_token_index]
            decoded_tokens = tokenizer.decode(distinct_tokens[gradient_token_index]).upper().split(" ")
            normalized_token_scores = unnormalized_token_scores/unnormalized_token_scores.sum()
            token_ranking = normalized_token_scores.argsort(descending=True)+1
            this_sequence_dict[key] = {
                'Sequence':decoded_sequence,
                'Prediction':pred_label,
                'Target':target_label,
                'Correct?':correct_prediction,
                'Distinct_Tokens':decoded_tokens,
                'Token_Importance':{k:v.item() for k,v in zip(decoded_tokens,normalized_token_scores)},
                'Token_Rank':{k:v.item() for k,v in zip(decoded_tokens,token_ranking)}
            }
            
        sequence_list.append(this_sequence_dict)

    return sequence_list

