import os
import csv
import pickle
import numpy as np
from statsmodels.regression.linear_model import burg
import lightgbm as lgb


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

    print('Calculating classification score for %s source...' % src_class)

    syn_data = syn_data.cpu().detach().numpy()
    syn_labels = syn_labels.cpu().detach().numpy()

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

    syn_features = extract_features(syn_data)
    trg_features = extract_features(trg_data)

    assert np.array_equal(np.unique(syn_doms), np.unique(trg_doms))
    assert np.array_equal(np.unique(syn_labels), np.unique(trg_labels))

    accs = []
    loglosses = []

    for domain in np.unique(syn_doms):
        syn_features_dom = syn_features[syn_doms == domain]
        trg_features_dom = trg_features[trg_doms == domain]

        syn_labels_dom = syn_labels[syn_doms == domain]
        trg_labels_dom = trg_labels[trg_doms == domain]
        
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(np.unique(syn_labels))}
        syn_labels_dom = np.array([label_mapping[x] for x in syn_labels_dom])
        trg_labels_dom = np.array([label_mapping[x] for x in trg_labels_dom])

        train_data = lgb.Dataset(syn_features_dom, label=syn_labels_dom)

        num_classes = len(np.unique(syn_labels_dom))

        params = {
            'objective': 'multiclass' if num_classes > 2 else 'binary',
            'num_class': num_classes if num_classes > 2 else 1,
            'metric': 'multi_logloss' if num_classes > 2 else 'binary_logloss',
            'seed': 2710,
            'verbosity': -1
        }

        model = lgb.train(params, train_data)

        preds = model.predict(trg_features_dom)
        eps = 1e-6
        preds = np.clip(preds, eps, 1 - eps)

        # Calculate accuracy and logloss
        if num_classes > 2:
            acc = np.mean(np.argmax(preds, axis=1) == trg_labels_dom)
            logloss = -np.mean(np.log(preds[np.arange(len(trg_labels_dom)), trg_labels_dom]))
        else:
            acc = np.mean((preds > 0.5).astype(int) == trg_labels_dom)
            logloss = -np.mean(trg_labels_dom * np.log(preds) + (1 - trg_labels_dom) * np.log(1 - preds))

        # print(f'Domain: {domain}, Accuracy: {acc:.4f}, Logloss: {logloss:.4f}')
        classification_scores = (acc, logloss)
        save_classification_scores(classification_scores, src_class, domain, step, mode, eval_dir, num_train_domains)

        accs.append(acc)
        loglosses.append(logloss)

    print(f'Mean accuracy: {np.mean(accs):.4f}, Mean logloss: {np.mean(loglosses):.4f}\n')

    return accs, loglosses


def safe_burg(x, order=4):
    if np.std(x) > 1e-6:  # Ensures there's enough variation in the data
        return burg(x, order)[0]
    else:
        return np.zeros(order)  # Return zeroed features if input data is constant
    

# Ensure no zero values in the entropy calculation
def entropy_safe(x):
    x_safe = np.clip(x, 1e-6, None)  # clip only lower bound
    return -np.sum(x_safe * np.log(x_safe), axis=2)


def extract_features_all(x):
    mean = np.mean(x, axis=2)
    std = np.std(x, axis=2)
    var = np.var(x, axis=2)
    min = np.min(x, axis=2)
    max = np.max(x, axis=2)
    thirdmoment = np.mean((x - np.mean(x, axis=2, keepdims=True))**3, axis=2)
    fourthmoment = np.mean((x - np.mean(x, axis=2, keepdims=True))**4, axis=2)
    skewness = thirdmoment / ((std+1e-6)**3)
    kurtosis = fourthmoment / ((std+1e-6)**4)
    mad = np.median(np.abs(x - np.median(x, axis=2, keepdims=True)), axis=2)
    sma = np.sum(np.abs(x), axis=2)
    energy = np.sum(x**2, axis=2)
    iqr = np.percentile(x, 75, axis=2) - np.percentile(x, 25, axis=2)
    firstquartile = np.percentile(x, 25, axis=2)
    secondquartile = np.percentile(x, 50, axis=2)
    thirdquartile = np.percentile(x, 75, axis=2)
    entropy = entropy_safe(x)
    autocorr_x = np.array([safe_burg(x[i, 0, :], order=4) for i in range(x.shape[0])])
    autocorr_y = np.array([burg(x[i, 1, :], order=4)[0] for i in range(x.shape[0])])
    autocorr_z = np.array([burg(x[i, 2, :], order=4)[0] for i in range(x.shape[0])])
    
    return np.concatenate([mean, std, var, min, max, thirdmoment, fourthmoment, 
                           skewness, kurtosis, mad, sma, energy, iqr, firstquartile, 
                           secondquartile, thirdquartile, entropy, 
                           autocorr_x, autocorr_y, autocorr_z], axis=1)

def extract_temporal_features(x):
    x = np.clip(x, 0, 1)
    return extract_features_all(x)


def extract_spectral_features(x):
    x_freq = np.fft.rfft(x, axis=2)
    x_mag = np.abs(x_freq)
    return extract_features_all(x_mag)


def extract_features(x):
    x_temporal = extract_temporal_features(x)
    x_spectral = extract_spectral_features(x)
    return np.concatenate([x_temporal, x_spectral], axis=1)



def save_classification_scores(classification_scores, src_class, domain, step, mode, eval_dir, num_train_domains):
    # Ensure the directory exists
    os.makedirs(eval_dir, exist_ok=True)
    # Path to the CSV file
    file_path = os.path.join(eval_dir, 'classification_scores.csv')
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