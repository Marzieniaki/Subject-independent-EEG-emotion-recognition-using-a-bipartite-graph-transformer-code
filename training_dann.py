import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils_dann import load_data, train, evaluate
from dann import *

# 

# Set the seed for reproducibility
import random
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # Set seed for CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Set seed for GPU
        torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs (if applicable)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Directory containing your data
file_path = "./data_basicSEED.csv"

# Total number of participants
num_participants = 15

# Hyperparameters
batch_size = 16
channels = 62  
embed_dim = 5  
num_heads = 5
drop_p = 0.1   
num_layers = 1 
num_epochs = 1
learning_rate = 5e-4* (batch_size/16 ) #  1e-2* (batch_size/16 )  #   1e-2# 1e-2#  5e-4# 5e-4

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Initialize a list to store the accuracies
participant_accuracies = []


modelToRun = 3 # 1 norma, 2 for spatial, 3 for temporal
# Open a file to write all results
with open('all_participants_results.txt', 'a') as result_file:
    # Loop over each participant
    #result_file.write("adding entropy\n")

    for target_participant in range(1, num_participants + 1):
        
        #Set seeds for this specific user # for replication
        set_seed(42)

        print(f"\nProcessing Participant {target_participant}/{num_participants}")
        
        # Load data
        train_loader, eval_loader = load_data(file_path=file_path, subject=target_participant, batch_size=batch_size)

        # model = DomainAdaptationModel(num_layers=num_layers,
        #                     embed_dim=embed_dim,
        #                     num_heads=num_heads,
        #                     expansion=2,
        #                     drop_p=drop_p,
        #                     )#.to(device)

        
        
        # Initialize the model
        if modelToRun==1:
            model = DomainAdaptationModel(num_layers=num_layers,
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            expansion=2,
                            drop_p=drop_p,
                            )
        elif  modelToRun==2:
        
            # spatial
            model = DomainAdaptationModel_spatial(num_layers=num_layers,
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            expansion=2,
                            drop_p=drop_p,
                            )
        
        elif  modelToRun==3:
            model = DomainAdaptationModel_temporal(num_layers=num_layers,
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            expansion=2,
                            drop_p=drop_p,
                            )
        
        
        model = nn.DataParallel(model)
        
        model.to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        

        optimizer = optim.AdamW(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-4
                                  )

        # optimizer = optim.SGD(model.parameters(),
        #                          lr=learning_rate,
        #                          momentum= 0.9,
        #                           weight_decay=0.01
        #                           )

        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, 
        #                                                        T_max=num_epochs)  # Maximum number of iterations)


        # Train the model
        for epoch in range(num_epochs):
            print(f"Participant {target_participant} - Epoch {epoch + 1}/{num_epochs}")
            train_loss, domain_loss, train_accuracy, train_domain_accuracy = train(model,
                                                                                   train_loader, eval_loader,
                                                                                    criterion,
                                                                                    optimizer, device,
                                                                                     epoch, num_epochs)
            print(f"Epoch {epoch + 1}: Training Loss = {train_loss:.4f}, Domain Loss = {domain_loss:.4f}, Training Accuracy = {train_accuracy:.4f}, Domain Accuracy = {train_domain_accuracy:.4f}")
        
            # Evaluate the model
            eval_loss, eval_accuracy, target_features, confmat, per_class_accuracy = evaluate(model,
                                                train_loader, eval_loader,
                                                criterion,
                                                optimizer, device,
                                                    epoch, num_epochs)
            print(f"Participant {target_participant}: Evaluation Loss = {eval_loss:.4f}, Evaluation Accuracy = {eval_accuracy:.4f}")
            #scheduler.step()
            # print(per_class_accuracy)

        # Store the accuracy
        participant_accuracies.append((target_participant, eval_accuracy))
        
        # Write the accuracy to the file
        result_file.write(f"Participant {target_participant} Test Accuracy: {eval_accuracy:.4f}\n")
        result_file.flush()


        

        np.savez(f"./model_features/{target_participant}.npz", features = target_features, confmat = confmat.cpu(), acc_per_class = per_class_accuracy)
        torch.save(model.state_dict(), f"./models_per_subject/{target_participant}.pth")

        del model
        torch.cuda.empty_cache()

# Optionally, print all accuracies
print("\nAll Participant Accuracies:")
for participant, accuracy in participant_accuracies:
    print(f"Participant {participant}: Accuracy = {accuracy:.4f}")
