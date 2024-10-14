import os
import csv
import numpy as np
import torch
import torch.nn as nn


class DomainClassifier(nn.Module):
    def __init__(self, num_timesteps=128, num_channels=3, num_domains=4, num_classes=5):
        super(DomainClassifier, self).__init__()
        # Shared layers as before
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

        # Prepare class-specific branches as a single module with conditionally applied outputs
        self.fc_class_branches = nn.Linear(100, 50 * num_classes)
        self.fc_final = nn.Linear(50, num_domains * num_classes)

    def forward(self, x, class_ids):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc_shared(x))

        # Process all class-specific branches simultaneously
        class_branches = self.fc_class_branches(x).view(x.size(0), -1, 50)
        class_outputs = class_branches[torch.arange(class_branches.size(0)), class_ids]

        # Final class-specific output
        final_outputs = self.fc_final(class_outputs.view(x.size(0), 50))
        return final_outputs.view(x.size(0), -1)
    

def calculate_domain_scores(domain_classifier, x_fake, y_trg, k_fake, src_class, trg_class):
    print(f'Calculating domain score for {src_class} -> {trg_class}...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out = domain_classifier(x_fake, y_trg)
    
    loss = nn.CrossEntropyLoss()
    loss_val = loss(out, torch.tensor(k_fake, dtype=torch.long).to(device)).item()

    preds = torch.argmax(out, dim=1).detach().cpu().numpy()
    accuracy = np.mean(preds == k_fake)

    print(f'Accuracy: {accuracy:.4f}, Loss: {loss_val:.4f}\n')

    return accuracy, loss_val



def save_domain_scores(domain_scores, src_class, trg_class, step, mode, eval_dir):
    # Ensure the directory exists
    os.makedirs(eval_dir, exist_ok=True)
    # Path to the CSV file
    file_path = os.path.join(eval_dir, 'domain_scores.csv')
    # Check if the file exists
    file_exists = os.path.exists(file_path)
    
    # Open the file in append mode if it exists, or write mode if it doesn't
    with open(file_path, mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)
        # If the file does not exist, write the header
        if not file_exists:
            writer.writerow(['step', 'mode', 'source', 'target', 'accuracy', 'loss'])
        
        accuracy, loss = domain_scores
        # Write the data rows
        writer.writerow([step, mode, src_class, trg_class, accuracy, loss])