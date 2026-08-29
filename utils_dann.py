import os
import re
import numpy as np
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch.autograd import Function
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"




import os
import re
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch.nn.functional as F

def entropy_minimization_loss(output):
    p = F.softmax(output, dim=1)  # Convert logits to probabilities
    log_p = F.log_softmax(output + 1e-10 , dim=1)
    batch_entropy = -torch.sum(p * log_p, dim=1)  # Calculate entropy for each sample in the batch
    return batch_entropy.mean()  # Return the mean entropy across the batch

import torch

def rbf_kernel(x, y, sigma):
    dist = torch.cdist(x, y) ** 2
    kernel = torch.exp(-dist / (2 * sigma ** 2))
    return kernel

def compute_mmd_loss(features_s, features_t, sigma_list):
    mmd_loss = 0
    for sigma in sigma_list:
        K_ss = rbf_kernel(features_s, features_s, sigma)
        K_tt = rbf_kernel(features_t, features_t, sigma)
        K_st = rbf_kernel(features_s, features_t, sigma)

        mmd = K_ss.mean() + K_tt.mean() - 2 * K_st.mean()
        mmd_loss += mmd

    return mmd_loss / len(sigma_list)

def load_data(file_path='./data_basicSEED.csv', subject=5, batch_size=16):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import TensorDataset, DataLoader
    import torch

    # Load the CSV file
    df = pd.read_csv(file_path)

    # Prepare windowsData and EEG columns
    windowsData = df[df['user'] == 10][['session', 'trial', 'window', 'label']].groupby(
        by=['label', 'session', 'trial']
    ).max().reset_index()
    eeg_cols = [col for col in df.columns if col not in ['user', 'session', 'trial', 'window', 'label']]

    max_windows = int(windowsData['window'].max()) + 1
    print('Max windows', max_windows )
    number_users = 15
    videos_per_user = 45
    feat_cols = 310

    # Initialize data arrays
    data = np.zeros([number_users * videos_per_user, max_windows, feat_cols])
    labels = np.empty(number_users * videos_per_user)
    users = np.empty(number_users * videos_per_user)

    unique_users = df['user'].unique()
    unique_session = df['session'].unique()
    unique_trials = df['trial'].unique()

    scaler = StandardScaler()

    for idx_user in range(len(unique_users)):
        current_user = unique_users[idx_user]
        data_user = df.loc[df['user'] == current_user, :]
        startIdx = idx_user * videos_per_user
        idx_video = 0

        for idx_session in range(len(unique_session)):
            for idx_trial in range(len(unique_trials)):
                data_video = data_user.loc[
                    (data_user['session'] == unique_session[idx_session]) &
                    (data_user['trial'] == unique_trials[idx_trial]), :
                ]

                # if data_video.empty:
                #     continue  # Skip if there's no data for this session and trial

                initialData = data_video[eeg_cols]
                dataNumpy = initialData.to_numpy().astype(float)
                normalized_data = scaler.fit_transform(dataNumpy)

                totalWinds = data_video.shape[0]

                if totalWinds == max_windows:
                    data[startIdx + idx_video] = normalized_data
                else:
                    data[startIdx + idx_video, :totalWinds] = normalized_data

                labels[startIdx + idx_video] = data_video.iloc[0]['label']
                users[startIdx + idx_video] = data_video.iloc[0]['user']

                idx_video += 1

                #Shyamal, what is this? all users have 45 videos
                # # Check if idx_video exceeds videos_per_user
                # if idx_video >= videos_per_user:
                #     break  # Prevent idx_video from exceeding allocated space

    # # Remove unused rows in data, labels, and users arrays # Shyamal, I did not get it? hy this --camilo
    # total_videos = idx_user * videos_per_user + idx_video
    # data = data[:total_videos]
    # labels = labels[:total_videos]
    # users = users[:total_videos]

    # Reshape data
    data = data.reshape(-1, max_windows, 62, 5)

    # Prepare target dataset
    target_tensor = np.reshape(data[users == subject], (-1, max_windows, 62*5))
    target_labels = labels[users == subject]
    target_labels[target_labels == -1] = 2  # Add this line

    target_tensor = torch.tensor(target_tensor, dtype=torch.float32)
    target_labels = torch.tensor(target_labels.astype(int), dtype=torch.long)
    target_dataset = TensorDataset(target_tensor, target_labels)
    target_dataloader = DataLoader(
        target_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    # Prepare source dataset
    source_tensor = np.reshape(data[users != subject], (-1, max_windows, 62*5))
    source_labels = labels[users != subject]
    source_labels[source_labels == -1] = 2  # Add this line

    source_tensor = torch.tensor(source_tensor, dtype=torch.float32)
    source_labels = torch.tensor(source_labels.astype(int), dtype=torch.long)
    source_dataset = TensorDataset(source_tensor, source_labels)
    source_dataloader = DataLoader(
        source_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    return source_dataloader, target_dataloader



def adjust_alpha(p, max_alpha=1.0):
   
    return max_alpha * ( 2. / (1. + np.exp(-10 * p)) - 1 )

 
def optimizer_scheduler(optimizer, p):
    for param_group in optimizer.param_groups:
        # initial learning rate 5e-4
        param_group['lr'] = 0.0005 / (1. + 10 * p) ** 0.75
    return optimizer                                      


def train(model, source_loader, target_loader, 
          criterion, optimizer, device, current_epoch, total_epochs):
    
    model.train()  # Set the model to training mode
    

    num_classes = 3
    total_task_loss = 0.0
    total_domain_loss = 0.0
    correct_task_predictions = 0
    correct_domain_predictions = 0
    total_samples = 0
    all_domain_samples = 0
    # Domain loss criterion

    # Create an iterator for the target_loader
    target_iter = iter(target_loader)

    # Adjust alpha for GRL if needed
    p = current_epoch/total_epochs
    alpha = adjust_alpha(p, max_alpha=.2 ) #if 'adjust_alpha' in globals() else 0.1
    optimizer = optimizer_scheduler(optimizer, p)

    steps_per_epoch = len(source_loader) 
    total_steps = total_epochs*steps_per_epoch # total across all iterations
    idx_step = 0
    for batch_idx, (source_data, source_labels) in enumerate(source_loader):
        source_data = source_data.to(device).float()
        source_labels = source_labels.to(device).long()
        # print(source_labels.shape)
        #current_step = ( idx_step + current_epoch*steps_per_epoch ) 
        #p = current_step/total_steps

        idx_step+=1

        #alpha = adjust_alpha(p, max_alpha=.2 ) #if 'adjust_alpha' in globals() else 0.1
        

        #print( f'epoch, alpha :{alpha} progress:{p}')

        #optimizer = optimizer_scheduler(optimizer, p)
        optimizer.zero_grad()

        

        # Get a batch from the target_loader
        try:
            target_data, _ = next(target_iter)

            if target_data.shape[0] < source_data.shape[0]:
                # Restart the target_loader if exhausted
                target_iter = iter(target_loader)
                target_data, _ = next(target_iter)

        except StopIteration:
            # Restart the target_loader if exhausted
            target_iter = iter(target_loader)
            target_data, _ = next(target_iter)

        if target_data.shape[0] > source_data.shape[0]:
            #print('aqui')
            target_data = target_data[ :source_data.shape[0] ]
            #print( target_data.shape ) 

        target_data = target_data.to(device).float()

        # Create key padding masks if necessary
        key_padding_mask_source_temporal = (source_data == 0).all(dim=-1).to(device)
        key_padding_mask_target_temporal = (target_data == 0).all(dim=-1).to(device)

      
        

        # Forward pass using both source and target data
        source_output, target_output, source_domain, target_domain, features_s, features_t = model(
            source_data,
            target_data,
            key_padding_mask_source_temporal,
            key_padding_mask_target_temporal,
            alpha
        )

        # Step 1: Calculate class frequencies in the current batch
        class_counts = torch.bincount(source_labels, minlength=num_classes)
        N = source_labels.size(0)

        # Step 2: Compute class weights (Inverse of class frequency)
        class_weights = N / (num_classes * class_counts.float() + 1e-6)  # +1e-6 to avoid division by zero
        #print(f'weights:{class_weights}')
          
        loss_fn_internal = torch.nn.CrossEntropyLoss(weight=class_weights)

        # 1.Classification loss
        task_loss = loss_fn_internal(source_output, source_labels)
        
        

        #print( 'domain shape', domain_pred.shape)

        domain_labels0 = torch.zeros(source_labels.size(0), dtype=torch.int64).to(device)
        domain_labels1 = torch.ones(target_data.size(0), dtype=torch.int64).to(device)
        domain_combined_label = torch.cat((domain_labels0, domain_labels1), dim=0)
        domain_pred = torch.cat((source_domain, target_domain), dim=0)


        domain_loss_s = criterion(source_domain, domain_labels0 )
        domain_loss_t = criterion(target_domain, domain_labels1 )

        entropy_loss = entropy_minimization_loss(target_output)


        sigma_list = [1, 2, 4, 8, 16]
        # # Entropy loss on target data
        
        # #entropy_loss = entropy_minimization_loss(target_output)  # Assuming this function is defined elsewhere
        mmd_loss = compute_mmd_loss(features_s, features_t, sigma_list)
        
        # # adding two losses
        lambda_entropy = 0.1      # You can adjust this weight as needed
        mmd_loss_weight = 0.1
        total_loss = task_loss + ( domain_loss_s + domain_loss_t )  + lambda_entropy*entropy_loss + mmd_loss * mmd_loss_weight
        total_loss.backward()



        


        # Create domain labels: 0 for source, 1 for target
        
        
        # Weights for losses
        #domain_loss_weight = 0.1  # You can adjust this weight as needed
        #lambda_entropy = 0.1      # You can adjust this weight as needed
        #mmd_loss_weight = 0.1
        # Total loss calculation
        #total_loss = task_loss + lambda_entropy * entropy_loss + domain_loss_weight * domain_loss + mmd_loss * mmd_loss_weight
        #total_loss = task_loss + alpha * ( Gd_loss_s + Gd_loss_t )  


        #max_norm = 0.5  # You can adjust this value as needed
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        #total_loss.backward()  # Backpropagate the total loss
        
        optimizer.step()  # Update weights

        # Update losses
        total_task_loss += task_loss.item()
        total_domain_loss += domain_loss_s.item() + domain_loss_t.item() #  Gd_loss_s.item() + Gd_loss_t.item()

        # Calculate task accuracy on source data
        predicted_task_labels = torch.argmax(torch.softmax(source_output, dim=1), dim=1)
        correct_task_predictions += (predicted_task_labels == source_labels).sum().item()

        # Calculate domain accuracy

        #domain_output = torch.cat([Gd_outputs_source, Gd_outputs_target], dim=0) 
        #domain_labels = torch.cat([domain_labels0, domain_labels1], dim=0) 
        

        predicted_domain_labels = torch.argmax(torch.softmax(domain_pred, dim=1), dim=1)
        correct_domain_predictions += (domain_combined_label == predicted_domain_labels).sum().item()

     
        total_samples += source_labels.size(0)
        # print(total_samples)
        all_domain_samples += ( domain_combined_label.size(0) )

        ## Print the updated learning rate
        #for param_group in optimizer.param_groups:
        #    print(f"Updated Learning Rate: {param_group['lr']}")

    avg_task_loss = total_task_loss / len(source_loader)
    avg_domain_loss = total_domain_loss / len(source_loader)
    train_task_accuracy = correct_task_predictions / total_samples
    train_domain_accuracy = correct_domain_predictions / (all_domain_samples)  # Since domain labels are for both source and target

    return avg_task_loss, avg_domain_loss, train_task_accuracy, train_domain_accuracy

from torchmetrics import ConfusionMatrix

def evaluate(model, source_loader, target_loader, 
          criterion, optimizer, device, current_epoch, total_epochs):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    confmat = ConfusionMatrix(task="multiclass", num_classes=3).to(device)

    target_feature_list = [ ]
    with torch.inference_mode():  # No gradient calculation needed
        # Create an iterator for the source_loader
        source_iter = iter(source_loader)

        for batch_idx, (target_data, target_labels) in enumerate(target_loader):
            target_data = target_data.to(device)
            target_labels = target_labels.to(device)

            target_data = target_data.float()
            target_labels = target_labels.long()

            try:
                source_data, _ = next(source_iter)
            except StopIteration:
                # Restart the source_loader if exhausted
                source_iter = iter(source_loader)
                source_data, _ = next(source_iter)

            if source_data.shape[0] > target_data.shape[0]:
                #print('aqui')
                source_data = target_data[ :target_data.shape[0] ]

            source_data = source_data.to(device)

            source_data = source_data.float()
            


            key_padding_mask_source_temporal = (source_data == 0).all(dim=-1)  # Checks along the embed_dim axis
            key_padding_mask_target_temporal = (target_data == 0).all(dim=-1)# Checks along the embed_dim axis
            
            #key_padding_mask_source_temporal = key_padding_mask_source_temporal.float()
            #key_padding_mask_target_temporal = key_padding_mask_target_temporal.float()

            #key_padding_mask_source_spatial = torch.reshape( key_padding_mask_source_temporal, (-1,1 ) ).repeat(1, 62)
            #key_padding_mask_target_spatial = torch.reshape( key_padding_mask_target_temporal, (-1,1 ) ).repeat(1, 62)

            #key_padding_mask_source_spatial = key_padding_mask_source_spatial.float()
            #key_padding_mask_target_spatial = key_padding_mask_target_spatial.float()

            # Forward pass using both source and target data
            _, target_output, _, _, _, target_features = model(
            source_data,
            target_data,
            key_padding_mask_source_temporal,
            key_padding_mask_target_temporal,
            0
             )

            
            target_feature_list.append( target_features.cpu() ) 

            # 1.Classification loss
            
            loss = criterion(target_output, target_labels)
                                    
           
            total_loss += loss.item()

            # Calculate accuracy
            predicted_labels = torch.argmax( torch.softmax( target_output, dim=1) , dim=1)  
            correct_predictions += (predicted_labels == target_labels).sum().item()
            total_samples += target_labels.size(0)
            confmat.update(predicted_labels, target_labels)
            
    final_confusion_matrix = confmat.compute()  # Compute the final confusion matrix
    per_class_accuracy = final_confusion_matrix.diag() / final_confusion_matrix.sum(1)
    per_class_accuracy = per_class_accuracy.cpu().numpy()  # Convert to numpy array if not already

    avg_loss = total_loss / len(target_loader)  # Calculate average loss
    accuracy = correct_predictions / total_samples  # Calculate accuracy
    return avg_loss, accuracy, np.concatenate(target_feature_list, axis=0), final_confusion_matrix, per_class_accuracy



# directory = "./data/EEG_DE_features/"
# target_participant = 1  
# source_dataloader, target_dataloader = load_data(directory, target_participant, batch_size=64)

# print(len(source_dataloader.dataset))
# print(len(target_dataloader.dataset))