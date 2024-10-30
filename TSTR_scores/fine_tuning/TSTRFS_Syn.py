import pickle
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
import os
import csv
import random
import copy

seed = 2710
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


config = {
    'realworld': {
        'dataset_name': 'realworld_128_3ch_4cl',
        'num_df_domains': 10,
        'num_dp_domains': 5,
        'num_classes': 4,
        'class_names': ['WAL', 'RUN', 'CLD', 'CLU'],
        'num_timesteps': 128,
        'num_channels': 3,
        'num_classes': 4,
    },
    'cwru': {
        'dataset_name': 'cwru_256_3ch_5cl',
        'num_df_domains': 4,
        'num_dp_domains': 4,
        'num_classes': 5,
        'class_names': ['IR', 'Ball', 'OR_centred', 'OR_orthogonal', 'OR_opposite'],
        'num_timesteps': 256,
        'num_channels': 3,
        'num_classes': 5,
    }
}




class TSTRClassifier(nn.Module):
    def __init__(self, 
                 num_timesteps=128, 
                 num_channels=3, 
                 num_classes=5, 
                 conv_channels=[16, 32, 64, 128], 
                 kernel_size=5, 
                 stride=1, 
                 pool_kernel_size=2, 
                 pool_stride=2, 
                 hidden_sizes=[100],
                 dropout_prob=0.25):
        super(TSTRClassifier, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        in_channels = num_channels
        for out_channels in conv_channels:
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2))
            self.bn_layers.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels
        
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # Calculate flattened input size for the first fully connected layer
        reduced_timesteps = num_timesteps
        for _ in range(len(conv_channels)):
            reduced_timesteps = (reduced_timesteps - 1) // pool_stride + 1  # Account for pooling effect
        
        fc_layers = []
        in_features = reduced_timesteps * conv_channels[-1]  # Initial input size for the first layer

        for hidden_size in hidden_sizes:
            fc_layers.append(nn.Linear(in_features, hidden_size))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(p=dropout_prob))
            in_features = hidden_size  # Update in_features for the next layer
        
        # Assign fully connected layers as a ModuleList
        self.fc_layers = nn.ModuleList(fc_layers)
        
        # Final classification layer
        self.fc_class = nn.Linear(in_features, num_classes)

    def forward(self, x):
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = self.pool(self.relu(bn(conv(x))))
        
        x = x.view(x.size(0), -1)  # Flatten
        
        # Pass through fully connected hidden layers
        for layer in self.fc_layers:
            x = layer(x)
        
        # Final output for class prediction
        class_outputs = self.fc_class(x)
        return class_outputs



def remap_labels(y):
    label_map = {clss: i for i, clss in enumerate(np.unique(y))}
    return np.array([label_map[clss] for clss in y])


def get_syn_data(dataset, src_class, domain=None):
    # Load configurations
    class_names = config[dataset]['class_names']

    try:
        with open(f'data/{dataset}_syndata_lat_fs.pkl', 'rb') as f:
            syndata_lat = pickle.load(f)
        with open(f'data/{dataset}_syndata_ref_fs.pkl', 'rb') as f:
            syndata_ref = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset files for {dataset} not found.")

    x = []
    y = []
    k = []

    for trg_class in class_names:
        if trg_class == src_class:
            continue
        x_lat, y_lat, k_lat = syndata_lat[(src_class, trg_class)]
        x.append(x_lat)
        y.append(y_lat)
        k.append(k_lat)
        x_ref, y_ref, k_ref = syndata_ref[(src_class, trg_class)]
        x.append(x_ref)
        y.append(y_ref)
        k.append(k_ref)

    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    k = np.concatenate(k, axis=0)

    if domain is not None:
        domain_mask = (k == domain)
        x, y, k = x[domain_mask], y[domain_mask], k[domain_mask]

    assert len(x) > 0, f"No data found"
    assert len(x) == len(y) == len(k), f"Data length mismatch"

    return x, y, k



def get_dp_data(dataset, src_class, domain=None):
    # Load configurations
    dataset_name = config[dataset]['dataset_name']
    class_idx = config[dataset]['class_names'].index(src_class)
    num_df_domains = config[dataset]['num_df_domains']

    try:
        with open(f'data/{dataset_name}.pkl', 'rb') as f:
            x, y, k = pickle.load(f)
        with open(f'data/{dataset_name}_fs.pkl', 'rb') as f:
            fs = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset files for {dataset} not found.")


    mask = (y != class_idx) & (k >= num_df_domains) & (fs == 0)

    # Apply initial mask
    x, y, k = x[mask], y[mask], k[mask]

    # Additional domain filtering if specified
    if domain is not None:
        domain_mask = (k == domain)
        x, y, k = x[domain_mask], y[domain_mask], k[domain_mask]

    assert len(x) > 0, f"No data found"
    assert len(x) == len(y) == len(k), f"Data length mismatch"

    return x, y, k



def get_dataloader(x, y, batch_size=64, shuffle=False):
    x = torch.tensor(x, dtype=torch.float32)
    y = remap_labels(y)
    y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader



def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

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



def train_model(model, train_loader, val_loader, optimizer, num_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    loss_train = []
    loss_val = []
    accuracy_val = []
    best_model_state = None
    best_loss = np.inf
    best_accuracy = 0

    # Set up linear learning rate decay
    lambda_lr = lambda epoch: 1 - epoch / num_epochs
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr)

    for epoch in range(num_epochs):
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

        val_accuracy, val_loss = evaluate_model(model, val_loader)
        if val_loss < best_loss:
            best_epoch = epoch
            best_accuracy = val_accuracy
            best_loss = val_loss
            best_model_state = model.state_dict().copy()

        loss_val.append(val_loss)
        accuracy_val.append(val_accuracy)

        current_lr = scheduler.get_last_lr()[0]
        if (epoch+1) % 10 == 0:
            print(f"\tEpoch {epoch + 1}/{num_epochs} - Train loss: {total_loss:.4f} - Val accuracy: {val_accuracy:.4f} - Val loss: {val_loss:.4f} - LR: {current_lr:.2e}")

    print(f"\tBest epoch: {best_epoch + 1} - Best val accuracy: {best_accuracy:.4f} - Best val loss: {best_loss:.4f}\n")

    # Load best model state
    model.load_state_dict(best_model_state)

    return model



def train_and_test(x_train, y_train, x_test, y_test, dataset, num_epochs=50, batch_size=64, 
                   learning_rate=0.0001, conv_channels=[16, 32, 64, 128], hidden_sizes=[100], dropout_prob=0.25):
    assert np.array_equal(np.unique(y_train), np.unique(y_test)), "Training and test labels do not match"

    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, shuffle=True, random_state=seed)

    train_loader = get_dataloader(x_tr, y_tr, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(x_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(x_test, y_test, batch_size=batch_size, shuffle=False)

    model = TSTRClassifier(num_timesteps=config[dataset]['num_timesteps'],
                           num_channels=config[dataset]['num_channels'],
                           num_classes=config[dataset]['num_classes']-1,
                           conv_channels=conv_channels,
                           hidden_sizes=hidden_sizes,
                           dropout_prob=dropout_prob)
    
    initial_lr = learning_rate
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)

    trained_model = train_model(model, train_loader, val_loader, optimizer, num_epochs)

    test_accuracy, test_loss = evaluate_model(trained_model, test_loader)

    return test_accuracy, test_loss



def save_scores(source, domain, accuracy, loss, name, dataset, **train_params):
    results_dir = 'results'
    # Ensure the directory exists
    os.makedirs(results_dir, exist_ok=True)
    # Path to the CSV file
    file_path = os.path.join(results_dir, f'{dataset}_{name}.csv')
    # Check if the file exists
    file_exists = os.path.exists(file_path)

    # Open the file in append mode if it exists, or write mode if it doesn't
    with open(file_path, mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)
        # If the file does not exist, write the header
        if not file_exists:
            writer.writerow(['source', 'domain', 'accuracy', 'loss', 'batch_size', 'learning_rate', 'conv_channels', 'hidden_sizes', 'dropout_prob'])
        # Write the data rows
        writer.writerow([source, domain, accuracy, loss, train_params['batch_size'], train_params['learning_rate'], 
                         train_params['conv_channels'], train_params['hidden_sizes'], train_params['dropout_prob']])




def compute_TSTRFS_Syn(dataset,                       
                       name,
                       num_epochs=50,
                       batch_size=64,
                       learning_rate=0.0001,
                       conv_channels=[16, 32, 64, 128], 
                       hidden_sizes=[100],
                       dropout_prob=0.25):

    # Collect parameters for train_and_test in a dictionary
    train_params = {
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'conv_channels': conv_channels,
        'hidden_sizes': hidden_sizes,
        'dropout_prob': dropout_prob,
        'batch_size': batch_size,
    }

    accs = []

    for src_class in config[dataset]['class_names']:
        print(f"Source class: {src_class}\n")

        for domain in range(config[dataset]['num_df_domains'], config[dataset]['num_df_domains'] + config[dataset]['num_dp_domains']):
            print(f"Domain: {domain}")

            # Load synthetic data
            x_syn_dom, y_syn_dom, k_syn_dom = get_syn_data(dataset, src_class, domain)
            print(f'x_syn_dom.shape: {x_syn_dom.shape} | np.unique(y_syn_dom): {np.unique(y_syn_dom)}')

            # Load Dp data
            x_dp_dom, y_dp_dom, k_dp_dom = get_dp_data(dataset, src_class, domain)
            print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)}\n')

            # Train on synthetic data and evaluate on Dp data
            print('Training on synthetic data...')
            acc_lat, loss_lat = train_and_test(x_syn_dom, y_syn_dom, x_dp_dom, y_dp_dom, dataset, **train_params)
            save_scores(src_class, domain, acc_lat, loss_lat, name, dataset, **train_params)
            accs.append(acc_lat)
            print(f'Source class: {src_class} | Domain: {domain} | Accuracy: {acc_lat:.4f} | Loss: {loss_lat:.4f}\n')

    print(f"Mean accuracy: {np.mean(accs):.4f}\n")



