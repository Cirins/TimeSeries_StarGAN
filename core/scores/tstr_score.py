import os
import csv
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim

class TSTRClassifier(nn.Module):
    def __init__(self, num_timesteps=128, num_channels=3, num_classes=5):
        super(TSTRClassifier, self).__init__()

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

        self.fc_class = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = self.relu(self.fc_shared(x))
        
        # Final output for class prediction
        class_outputs = self.fc_class(x)
        return class_outputs



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


def calculate_classification_scores(syn_data, syn_labels, syn_doms, src_class, trg_classes, step, mode, 
                                    eval_dir, class_names, dataset_name, num_train_domains):

    print('Calculating TSTR score for %s source...\n' % src_class)

    classes_dict = {clss: i for i, clss in enumerate(class_names)}

    trg_data = []
    trg_labels = []
    trg_doms = []   

    for trg_class in trg_classes:

        trg_idx = classes_dict[trg_class]
        x_trg, y_trg, k_trg = get_data(dataset_name, trg_idx, num_train_domains)

        trg_data.append(x_trg)
        trg_labels.append(y_trg)
        trg_doms.append(k_trg)

    trg_data = np.concatenate(trg_data, axis=0)
    trg_labels = np.concatenate(trg_labels, axis=0)
    trg_doms = np.concatenate(trg_doms, axis=0)

    assert np.array_equal(np.unique(syn_doms), np.unique(trg_doms))
    assert np.array_equal(np.unique(syn_labels), np.unique(trg_labels))

    accs = []
    loglosses = []

    for domain in np.unique(syn_doms):
        syn_data_dom = syn_data[syn_doms == domain]
        trg_data_dom = trg_data[trg_doms == domain]

        syn_labels_dom = syn_labels[syn_doms == domain]
        trg_labels_dom = trg_labels[trg_doms == domain]

        print(f'\nDomain: {domain}, Source: {src_class}, Target: {trg_classes}, Syn data: {syn_data_dom.shape}, Trg data: {trg_data_dom.shape}')
        
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(np.unique(syn_labels))}
        syn_labels_dom = np.array([label_mapping[x] for x in syn_labels_dom])
        trg_labels_dom = np.array([label_mapping[x] for x in trg_labels_dom])

        acc, logloss = compute_accuracy(syn_data_dom, syn_labels_dom, trg_data_dom, trg_labels_dom)

        print(f'Domain: {domain}, Accuracy: {acc:.4f}, Logloss: {logloss:.4f}')
        classification_scores = (acc, logloss)
        save_classification_scores(classification_scores, src_class, domain, step, mode, eval_dir, num_train_domains)

        accs.append(acc)
        loglosses.append(logloss)

    print(f'\nMean accuracy: {np.mean(accs):.4f}, Mean logloss: {np.mean(loglosses):.4f}\n\n')

    return accs, loglosses



def compute_accuracy(x_train, y_train, x_test, y_test):
    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=2710, stratify=y_train, shuffle=True)
    tr_loader, val_loader, test_loader = setup_training(x_tr, y_tr, x_val, y_val, x_test, y_test, batch_size=64)
    
    model = TSTRClassifier(num_timesteps=x_train.shape[2], num_channels=x_train.shape[1], num_classes=len(np.unique(y_train)))
    loss_fn = nn.CrossEntropyLoss()
    initial_lr = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)

    best_model_state = train_model(model, tr_loader, val_loader, loss_fn, optimizer, epochs=50)
    best_model = TSTRClassifier(num_timesteps=x_train.shape[2], num_channels=x_train.shape[1], num_classes=len(np.unique(y_train)))
    best_model.load_state_dict(best_model_state)
    test_accuracy, test_loss = evaluate_model(best_model, test_loader, loss_fn)

    return test_accuracy, test_loss


def setup_training(x_tr, y_tr, x_val, y_val, x_test, y_test, batch_size=64):
    # Convert numpy arrays to torch tensors
    x_train_tensor = torch.tensor(x_tr, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_tr, dtype=torch.long)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create datasets and loaders
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs=300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_train = []
    loss_val = []
    accuracy_val = []
    best_loss = np.inf
    best_accuracy = 0

    # Set up linear learning rate decay
    lambda_lr = lambda epoch: 1 - epoch / epochs
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(train_loader)
        loss_train.append(total_loss)

        # Update learning rate
        scheduler.step()

        val_accuracy, val_loss = evaluate_model(model, val_loader, loss_fn)
        if val_accuracy > best_accuracy:
            best_epoch = epoch
            best_accuracy = val_accuracy
            best_loss = val_loss
            best_model_state = model.state_dict().copy()
        loss_val.append(val_loss)
        accuracy_val.append(val_accuracy)

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}/{epochs} - Train loss: {total_loss:.4f} - Val loss: {val_loss:.4f} - Val accuracy: {val_accuracy:.4f} - LR: {current_lr:.6f}")
    
    print(f"Best epoch: {best_epoch + 1} - Best val accuracy: {best_accuracy:.4f} - Best val loss: {best_loss:.4f}")

    return best_model_state


def evaluate_model(model, test_loader, loss_fn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)
            total_loss += loss.item()

            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == y_batch).sum().item()
            total_predictions += len(y_batch)

    total_loss /= len(test_loader)
    accuracy = correct_predictions / total_predictions

    return accuracy, total_loss
    



def save_classification_scores(classification_scores, src_class, domain, step, mode, eval_dir, num_train_domains):
    # Ensure the directory exists
    os.makedirs(eval_dir, exist_ok=True)
    # Path to the CSV file
    file_path = os.path.join(eval_dir, 'TSTR_scores.csv')
    # Check if the file exists
    file_exists = os.path.exists(file_path)
    
    # Open the file in append mode if it exists, or write mode if it doesn't
    with open(file_path, mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)
        # If the file does not exist, write the header
        if not file_exists:
            writer.writerow(['step', 'mode', 'source', 'domain', 'accuracy', 'loss'])

        accuracy, loss = classification_scores
        # Write the data rows
        writer.writerow([step, mode, src_class, domain+num_train_domains, accuracy, loss])