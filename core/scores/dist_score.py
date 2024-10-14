import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import csv
import numpy as np


def get_data(dataset_name, class_idx, num_train_domains):

    # Load the dataset
    with open(f'data/{dataset_name}.pkl', 'rb') as f:
        x, y, k = pickle.load(f)

    with open(f'data/{dataset_name}_fs.pkl', 'rb') as f:
        fs = pickle.load(f)

    # Filter out the samples that are used for finetuning
    x = x[fs == 0]
    y = y[fs == 0]
    k = k[fs == 0]
    
    x_ = x[(y == class_idx) & (k >= num_train_domains)]
    y_ = y[(y == class_idx) & (k >= num_train_domains)]
    k_ = k[(y == class_idx) & (k >= num_train_domains)] - num_train_domains

    return x_, y_, k_


class SiameseNet(nn.Module):
    def __init__(self, num_channels=3, num_classes=5, num_timesteps=128):
        super(SiameseNet, self).__init__()
        # Shared layers
        self.conv1 = nn.Conv1d(num_channels, 16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)
        self.fc_shared = nn.Linear(num_timesteps * 8, 100)

        # Class-specific branches
        self.fc_class_branches = nn.Linear(100, 50 * num_classes)

    def forward_once(self, x, class_id):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc_shared(x))

        # Process class-specific branch
        class_branches = self.fc_class_branches(x).view(x.size(0), -1, 50)
        class_output = class_branches[torch.arange(class_branches.size(0)), class_id]
        return class_output

    def forward(self, input1, input2, class_id1, class_id2):
        output1 = self.forward_once(input1, class_id1)
        output2 = self.forward_once(input2, class_id2)
        return output1, output2
    

def contrastive_loss(output1, output2, label, margin=1.0):
    euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
    loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                  (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss_contrastive, euclidean_distance


def calculate_dist_scores(siamese_net_te, x_fake, y_trg, k_fake, src_class, trg_class, dataset_name, num_train_domains, class_names):
    print(f'Calculating distance score for {src_class} -> {trg_class}...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classes_dict = {clss: i for i, clss in enumerate(class_names)}
    trg_idx = classes_dict[trg_class]

    x_real, y_real, k_real = get_data(dataset_name, trg_idx, num_train_domains)
    
    # Ensure k_real and k_fake contain the same set of unique elements
    assert np.array_equal(np.unique(k_real), np.unique(k_fake)), 'k_real and k_fake contain different unique elements'

    siamese_net_te.eval()
    with torch.no_grad():
        real_features = siamese_net_te.forward_once(torch.tensor(x_real, dtype=torch.float32).to(device), 
                                                    torch.tensor(y_real, dtype=torch.long).to(device))
        fake_features = siamese_net_te.forward_once(x_fake.clone().detach().to(device).float(),
                                                    y_trg.clone().detach().to(device).long())
    
    # Calculate average distance for each unique k
    unique_k = np.unique(k_real)
    avg_distances = {}
    for k in unique_k:
        real_indices = [i for i, val in enumerate(k_real) if val == k]
        fake_indices = [i for i, val in enumerate(k_fake) if val == k]
        
        real_k_features = real_features[real_indices]
        fake_k_features = fake_features[fake_indices]

        real_k_features_exp = real_k_features.unsqueeze(1)
        fake_k_features_exp = fake_k_features.unsqueeze(0)
        
        distances = F.pairwise_distance(real_k_features_exp, fake_k_features_exp, keepdim=True)
        avg_distance = distances.mean().item()
        avg_distances[k] = avg_distance

    avg_avg_distance = np.mean(list(avg_distances.values()))
    print(f'Average distance: {avg_avg_distance}\n')

    return avg_distances



def save_dist_scores(dist_scores, src_class, trg_class, step, mode, eval_dir):
    # Ensure the directory exists
    os.makedirs(eval_dir, exist_ok=True)
    # Path to the CSV file
    file_path = os.path.join(eval_dir, 'dist_scores.csv')
    # Check if the file exists
    file_exists = os.path.exists(file_path)
    
    # Open the file in append mode if it exists, or write mode if it doesn't
    with open(file_path, mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)
        # If the file does not exist, write the header
        if not file_exists:
            writer.writerow(['step', 'mode', 'source', 'target', 'domain', 'distance'])
            
        # Write the data rows
        for k, distance in dist_scores.items():
            writer.writerow([step, mode, src_class, trg_class, k, distance])
