import torch
from tqdm import tqdm
from datetime import datetime
import pickle
import os
from model_settings import settings_dict

def combined_loss(outputs,targets,weights):
    loss = 0
    for key in targets.keys():
        loss += torch.nn.functional.cross_entropy(outputs[key],targets[key])*weights[key]
    return loss

def model_inference(model,data,weights,dimensions,checkpointloc):
    inf_start = datetime.now().strftime('%d-%m-%Y_%H%M')
    if checkpointloc is not None:
        checkpoint = torch.load(checkpointloc)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()


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
    
    return weights

def model_val(model,val_loader,weights,dimensions,logging,loggingfolder,device,train_start):
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
    for key in weights:
        avg_accuracy_overall[key] = torch.tensor(run_accuracy[key]).mean().item()
    avg_loss_overall = torch.tensor(run_loss).mean().item()
    return avg_accuracy_overall, avg_loss_overall

def model_train_loop(epochs,model,optimizer,train_loader,val_loader,weights,dimensions,accstop,logging,loggingfolder,checkpointloc,device):
    epoch_stats = {}
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
        weights = model_train(epoch,model,optimizer,train_loader,weights,dimensions,logging,loggingfolder,device,train_start,checkpointloc)
        model.eval()
        print('Validation run on epoch {:02}'.format(epoch))
        val_accuracy, val_loss = model_val(model,val_loader,weights,dimensions,logging,loggingfolder,device,train_start)
        print('Validation accuracy: {:.4f}, loss: {:.4f}'.format(val_accuracy,val_loss))
        torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
            },'checkpoints/{}/epoch{:02}'.format(train_start,epoch))
        acc_flag = True
        for key in weights:
            if val_accuracy[key] < accstop:
                acc_flag = False
        if acc_flag:
            break # accuracy threshold is reached
