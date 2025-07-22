import numpy as np
import pandas as pd
import pickle
import random
import cv2
import os
from collections import Counter
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import glob
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score
import argparse
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from torch.cuda.amp import autocast

# Function to extract labels from filenames
def extract_labels_from_filename(filename):
    base = os.path.basename(filename)
    parts = base.split('_')
    age = int(parts[0])
    gender = int(parts[1])
    race = int(parts[2].split('.')[0])
    return age, gender, race

# Load UTKFace dataset with age as sensitive attribute and gender as classification target
def load_utkface(paths, verbose=-1):
    data = []
    labels = []
    sensitive_attrs = []  # Age as sensitive attribute
    for (i, imgpath) in enumerate(paths):
        # Load RGB image
        im_rgb = cv2.imread(imgpath)  # Load in BGR format by default
        im_rgb = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        im_rgb = cv2.resize(im_rgb, (64, 64))  # Resize to (64, 64) for CNN
        image = np.array(im_rgb) / 255.0  # Normalize pixel values to [0, 1]

        try:
            # Extract age, gender, and race from the filename
            age, gender, race = extract_labels_from_filename(imgpath)

            # Append image data and gender as target
            data.append(image)
            labels.append(gender)  # Gender as classification target (0 or 1)

            # Encode age into 2 bins (sensitive attribute)
            if age <= 50:
                sensitive_attrs.append(0)  # Bin 0 
            else:
                sensitive_attrs.append(1)  # Bin 1

        except Exception as e:
            continue

        # Print progress if verbose is enabled
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print(f"[INFO] Processed {i + 1}/{len(paths)} images")

    # Return images, gender labels, and sensitive attribute (age)
    return np.array(data), np.array(labels), np.array(sensitive_attrs)

from imblearn.over_sampling import SMOTE

def G_SM(X, y, n_to_sample, cl):
    # determining the number of samples to generate
    #n_to_sample = 10

    # fitting the model
    n_neigh = 5 + 1
    nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    nn.fit(X)
    dist, ind = nn.kneighbors(X)

    # generating samples
    base_indices = np.random.choice(list(range(len(X))), n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neigh)), n_to_sample)

    X_base = X[base_indices]
    X_neighbor = X[ind[base_indices, neighbor_indices]]

    samples = X_base + np.multiply(np.random.rand(n_to_sample, 1),
                                    X_neighbor - X_base)

    del X, y, dist, ind, X_base, X_neighbor
    torch.cuda.empty_cache()

    # Use `cl` as the label for all synthetic samples
    return samples, [cl] * n_to_sample

# DeepSMOTE Encoder/Decoder
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.n_channel = args['n_channel']  # 3 for UTKFace
        self.dim_h = args['dim_h']          # Hidden layer size (64)
        self.n_z = args['n_z']              # Latent dimension (300)

        print(f"Encoder: dim_h={self.dim_h}, n_channel={self.n_channel}, n_z={self.n_z}")

        # Encoder: Convolution layers to extract features
        self.conv = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),  # 64x64 -> 32x32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),  # 32x32 -> 16x16
            nn.BatchNorm2d(self.dim_h * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),  # 16x16 -> 8x8
            nn.BatchNorm2d(self.dim_h * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),  # 8x8 -> 4x4
            nn.BatchNorm2d(self.dim_h * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Fully connected layer for latent space (4x4x512 -> 300)
        self.fc = nn.Linear(self.dim_h * 8 * 4 * 4, self.n_z)

    def forward(self, x):
        # Pass through conv layers
        print(f"Shape in Encoder: {x.shape}")
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.n_channel = args['n_channel']  # 3 for UTKFace
        self.dim_h = args['dim_h']          # Hidden layer size (64)
        self.n_z = args['n_z']              # Latent dimension (300)

        print(f"Decoder: dim_h={self.dim_h}, n_channel={self.n_channel}, n_z={self.n_z}")

        # Fully connected layer to reshape latent space to feature map
        self.fc = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 4 * 4),
            nn.ReLU()
        )

        # Deconvolutional layers to reconstruct the image
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4, 2, 1, bias=False),  # 4x4 -> 8x8
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4, 2, 1, bias=False),  # 8x8 -> 16x16
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.dim_h * 2, self.dim_h, 4, 2, 1, bias=False),  # 16x16 -> 32x32
            nn.BatchNorm2d(self.dim_h),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.dim_h, self.n_channel, 4, 2, 1, bias=False),  # 32x32 -> 64x64
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.dim_h * 8, 4, 4)
        x = self.deconv(x)
        return x

def apply_deepsmote(data, labels, age_bins, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Shape entering DeepSMOTE: {data.shape}")

    encoded_dim = args['n_z']
    encoder = Encoder(args).to(device)
    decoder = Decoder(args).to(device)

    encoder.eval()
    decoder.eval()

    data_tensor = torch.Tensor(data).to(device)
    with torch.no_grad():
        latent_space = encoder(data_tensor)

    synthetic_samples, synthetic_labels, synthetic_ages = [], [], []

    age_groups, counts = np.unique(age_bins, return_counts=True)
    max_count = np.max(counts)
    for age_group in age_groups:
        group_indices = np.where(age_bins == age_group)[0]
        n_current = len(group_indices)
        n_to_sample = max_count - n_current
        if n_to_sample <= 0:
            continue
        X_group = latent_space[group_indices].cpu().numpy()
        samples, _ = G_SM(X_group, labels[group_indices], n_to_sample, labels[group_indices][0])
        samples_tensor = torch.Tensor(samples).to(device)
        with torch.no_grad():
            decoded_samples = decoder(samples_tensor).cpu().numpy()
        synthetic_samples.append(decoded_samples)
        synthetic_labels.extend(np.random.choice(labels[group_indices], n_to_sample, replace=True).tolist())
        synthetic_ages.extend([age_group] * n_to_sample)

    print(f"Shape of original labels: {labels.shape}")
    print(f"Shape of synthetic labels: {np.array(synthetic_labels).shape}")

    if len(synthetic_samples) > 0:
        synthetic_samples = np.vstack(synthetic_samples)
        torch.cuda.empty_cache()
        data_smote = np.vstack([data, synthetic_samples])

        if len(labels.shape) > 1 and labels.shape[1] > 1:
            labels_flattened = np.argmax(labels, axis=1)
        else:
            labels_flattened = labels

        labels_smote = np.concatenate([labels_flattened, synthetic_labels])
        age_smote = np.concatenate([age_bins, synthetic_ages])
    else:
        data_smote, labels_smote, age_smote = data, labels, age_bins

    del encoder, decoder
    torch.cuda.empty_cache()

    return data_smote, labels_smote, age_smote


# Federated Learning utilities

def scale_weights(state_dict, scalar):
    return {k: v * scalar for k, v in state_dict.items()}

def sum_weights(weight_list):
    avg_weights = {}
    for key in weight_list[0].keys():
        avg_weights[key] = sum([client_weights[key] for client_weights in weight_list])
    return avg_weights

def compute_alpha(client_loader, sensitive_attr_index=0, num_classes=2):
    """
    Compute the alpha parameter for a client based on the sensitive attribute.

    Args:
        client_loader: DataLoader for the client's training data.
        sensitive_attr_index: Index of the sensitive attribute in the dataset (default: 0 for age).
        num_classes: Number of classes (default: 2).

    Returns:
        Alpha value for the client.
    """
    # Count the number of samples and minority samples
    total_samples = len(client_loader.dataset)
    sensitive_attrs = [age.item() for _, _, age in client_loader.dataset]
    minority_samples = min(Counter(sensitive_attrs).values())  # Count samples in the minority group

    # Compute alpha
    if minority_samples > 0:
        alpha = total_samples / (minority_samples * num_classes)
    else:
        alpha = 1.0  # Default to 1 if no minority samples exist

    return alpha


def reweighted_fedavg(scaled_weights, client_loaders, num_classes=2):
    """
    Perform FedAvg with reweighting based on alpha values.

    Args:
        scaled_weights: List of scaled weights from clients.
        client_loaders: Dictionary of client DataLoaders.
        num_classes: Number of classes (default: 2).

    Returns:
        Reweighted average of client weights.
    """
    total_samples = sum(len(client['train_loader'].dataset) for client in client_loaders.values())
    reweighted_weights = {}

    # Initialize reweighted_weights with zeros
    for key in scaled_weights[0].keys():
        reweighted_weights[key] = torch.zeros_like(scaled_weights[0][key])

    # Compute reweighted average
    for client_id, client_weights in enumerate(scaled_weights):
        client_loader = client_loaders[f'client_{client_id}']['train_loader']
        alpha = compute_alpha(client_loader, num_classes=num_classes)
        for key in client_weights.keys():
            reweighted_weights[key] += alpha * client_weights[key].float()

    # Normalize by total samples
    for key in reweighted_weights.keys():
        reweighted_weights[key] /= total_samples

    return reweighted_weights

class ClientDataset(Dataset):
    def __init__(self, images, labels, ages, transform=None):
        self.images = images
        self.labels = labels
        self.ages = ages
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]  # shape: (64, 64, 3)
        label = self.labels[idx]
        age = self.ages[idx]

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Now shape: (3, 64, 64)

        if self.transform:
            image = self.transform(image)

        return image, label, age

# Assign data, labels (gender), and ages to clients and also split into train/val

def assign_clients(data, labels, ages, client_indices, validation_size=0.2, batch_size=32, transform=None, random_state=42):
    clients = {}
    for client_id, indices in client_indices.items():
        client_data = data[indices]
        client_labels = labels[indices]
        client_ages = ages[indices]
        
        # Use a deterministic random state for each client, derived from the main random_state
        # but made unique for each client_id
        client_random_state = random_state + int(client_id) * 100
        
        X_train, X_val, y_train, y_val, age_train, age_val = train_test_split(
            client_data, client_labels, client_ages,
            test_size=validation_size, 
            random_state=client_random_state,  # Use the client-specific random state
            stratify=client_labels
        )
        
        train_dataset = ClientDataset(X_train, y_train, age_train, transform=transform)
        val_dataset = ClientDataset(X_val, y_val, age_val, transform=transform)
        
        # For reproducibility in DataLoader, set the worker_init_fn and generator
        g = torch.Generator()
        g.manual_seed(client_random_state)
        
        clients[f'client_{client_id}'] = {
            'train_loader': DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                worker_init_fn=lambda _: np.random.seed(client_random_state),
                generator=g
            ),
            'val_loader': DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False
            )
        }
    return clients

def create_balanced_client_partition(X_train, Y_train, age_train, num_clients, alpha, seed=42, min_samples_per_group=5):
    """
    Create a balanced client partition ensuring each client has a minimum number of samples
    from each demographic group (combination of gender and age).
    
    Args:
        X_train: Training data
        Y_train: Gender labels
        age_train: Age group labels
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter
        seed: Random seed
        min_samples_per_group: Minimum number of samples per demographic group per client
        
    Returns:
        Dictionary mapping client IDs to indices of their assigned samples
    """
    # Set a fixed random seed
    partition_seed = int(seed * 100 + alpha * 10)
    rng = np.random.RandomState(partition_seed)
    
    # Identify all demographic groups (combinations of gender and age)
    groups = {}
    for gender in [0, 1]:  # 0: male, 1: female
        for age in [0, 1]:  # 0: under 50, 1: 50 and above
            mask = (Y_train == gender) & (age_train == age)
            indices = np.where(mask)[0]
            if len(indices) > 0:
                groups[(gender, age)] = indices
    
    client_indices = {i: [] for i in range(num_clients)}
    
    # First, ensure minimum samples per demographic group per client
    for group_key, group_indices in groups.items():
        # If we don't have enough samples to guarantee min_samples for all clients, reduce as needed
        available_samples = len(group_indices)
        samples_per_client = min(min_samples_per_group, available_samples // num_clients)
        
        if samples_per_client == 0:
            print(f"WARNING: Not enough samples for group {group_key}. Only {available_samples} available for {num_clients} clients.")
            # Distribute available samples as evenly as possible
            samples_per_client = max(1, available_samples // num_clients)
        
        # Shuffle indices for this demographic group
        rng.shuffle(group_indices)
        
        # Distribute minimum samples to each client
        for client_id in range(num_clients):
            if client_id < len(group_indices) // samples_per_client:
                start_idx = client_id * samples_per_client
                end_idx = start_idx + samples_per_client
                client_indices[client_id].extend(group_indices[start_idx:end_idx])
    
    # Then, distribute remaining samples using Dirichlet distribution
    remaining_indices = []
    for group_indices in groups.values():
        # Find indices that haven't been assigned yet
        assigned = set()
        for client_id in range(num_clients):
            assigned.update(client_indices[client_id])
        
        unassigned = [idx for idx in group_indices if idx not in assigned]
        remaining_indices.extend(unassigned)
    
    # Apply Dirichlet partitioning to remaining indices
    if len(remaining_indices) > 0:
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.array(proportions * len(remaining_indices), dtype=int)
        
        # Adjust proportions to match available samples
        while proportions.sum() > len(remaining_indices):
            proportions[np.argmax(proportions)] -= 1
        while proportions.sum() < len(remaining_indices):
            proportions[np.argmin(proportions)] += 1
        
        # Shuffle remaining indices
        rng.shuffle(remaining_indices)
        
        # Split indices and distribute to clients
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + proportions[client_id]
            client_indices[client_id].extend(remaining_indices[start_idx:end_idx])
            start_idx = end_idx
    
    # Print distribution statistics
    print("\n=== Client Data Distribution Statistics ===")
    for client_id, indices in client_indices.items():
        client_y = Y_train[indices]
        client_age = age_train[indices]
        
        print(f"Client {client_id} - Total samples: {len(indices)}")
        for gender in [0, 1]:
            for age in [0, 1]:
                count = np.sum((client_y == gender) & (client_age == age))
                print(f"  Gender {gender}, Age {age}: {count} samples")
    
    return client_indices

class BalancedClientDataset(Dataset):
    """
    Dataset that ensures balanced sampling across demographic groups during training.
    """
    def __init__(self, images, labels, ages, transform=None):
        self.images = images
        self.labels = labels
        self.ages = ages
        self.transform = transform
        
        # Create demographic group identifiers (combination of gender and age)
        self.groups = []
        for gender, age in zip(labels, ages):
            self.groups.append((int(gender), int(age)))
        
        # Calculate sampling weights to balance demographic groups
        self.weights = self._calculate_weights()
        
    def _calculate_weights(self):
        """Calculate weights for balanced sampling across demographic groups."""
        # Count samples in each group
        group_counts = {}
        for group in self.groups:
            if group not in group_counts:
                group_counts[group] = 0
            group_counts[group] += 1
        
        # Compute weights (inverse of group frequency)
        weights = []
        for group in self.groups:
            if group_counts[group] > 0:
                weight = 1.0 / group_counts[group]
            else:
                weight = 0.0
            weights.append(weight)
        
        return weights

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]  # shape: (64, 64, 3)
        label = self.labels[idx]
        age = self.ages[idx]

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Now shape: (3, 64, 64)

        if self.transform:
            image = self.transform(image)

        return image, label, age

def dirichlet_partition(data, age_bins, gender_labels, num_clients, alpha, seed=42, min_samples_per_group=5):
    """
    Partition data using a Dirichlet distribution to create heterogeneous distributions
    among clients, while ensuring a minimum number of samples per demographic group per client.
    
    Args:
        data: The input data to partition
        age_bins: Age group labels (sensitive attribute)
        gender_labels: Gender labels (target variable)
        num_clients: Number of clients to distribute data to
        alpha: Dirichlet concentration parameter (lower = more heterogeneous)
        seed: Random seed for reproducibility
        min_samples_per_group: Minimum samples per demographic group per client
        
    Returns:
        Dictionary mapping client IDs to indices of their assigned samples
    """
    # Set a fixed random seed based on alpha and the main seed
    partition_seed = int(seed * 100 + alpha * 10)
    rng = np.random.RandomState(partition_seed)
    
    # Create demographic groups (combinations of gender and age)
    client_indices = {i: [] for i in range(num_clients)}
    
    # Get unique demographic groups (age, gender combinations)
    demographic_groups = []
    for gender in [0, 1]:  # 0: male, 1: female
        for age in [0, 1]:  # 0: under 50, 1: 50 and above
            demographic_groups.append((age, gender))
    
    # First, ensure minimum samples per demographic group per client
    for age_group, gender_group in demographic_groups:
        # Get indices for current demographic group
        group_indices = np.where((age_bins == age_group) & (gender_labels == gender_group))[0]
        
        if len(group_indices) == 0:
            continue
            
        # Shuffle indices for this demographic group
        rng.shuffle(group_indices)
        
        # Calculate samples_per_client based on available samples and minimum requirement
        available_samples = len(group_indices)
        samples_per_client = min(min_samples_per_group, available_samples // num_clients)
        
        if samples_per_client == 0:
            print(f"WARNING: Not enough samples for group (Age={age_group}, Gender={gender_group}). " 
                  f"Only {available_samples} available for {num_clients} clients.")
            # Distribute what we can
            samples_per_client = max(1, available_samples // num_clients)
        
        # Distribute minimum samples to each client
        for client_id in range(num_clients):
            if client_id < len(group_indices) // samples_per_client:
                start_idx = client_id * samples_per_client
                end_idx = start_idx + samples_per_client
                client_indices[client_id].extend(group_indices[start_idx:end_idx])
        
        # Remove the already assigned indices
        total_assigned = samples_per_client * min(num_clients, len(group_indices) // samples_per_client)
        remaining_indices = group_indices[total_assigned:]
        
        # Skip to next group if no remaining indices
        if len(remaining_indices) == 0:
            continue
            
        # Now apply Dirichlet partitioning on the remaining indices
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.array(proportions * len(remaining_indices), dtype=int)
        
        # Adjust proportions to match available samples
        while proportions.sum() > len(remaining_indices):
            proportions[np.argmax(proportions)] -= 1
        while proportions.sum() < len(remaining_indices):
            proportions[np.argmin(proportions)] += 1
        
        # Split remaining indices and distribute to clients
        start_idx = 0
        for i in range(num_clients):
            end_idx = start_idx + proportions[i]
            client_indices[i].extend(remaining_indices[start_idx:end_idx])
            start_idx = end_idx
    
    # Print distribution statistics
    print("\n=== Client Data Distribution Statistics ===")
    for client_id, indices in client_indices.items():
        client_y = gender_labels[indices]
        client_age = age_bins[indices]
        
        print(f"Client {client_id} - Total samples: {len(indices)}")
        for gender in [0, 1]:
            for age in [0, 1]:
                count = np.sum((client_y == gender) & (client_age == age))
                print(f"  Gender {gender}, Age {age}: {count} samples")
    
    return client_indices
    
def assign_clients_with_balanced_sampling(data, labels, ages, client_indices, 
                                         validation_size=0.2, batch_size=32, 
                                         transform=None, random_state=42):
    """
    Assign data to clients and create balanced samplers for training.
    """
    clients = {}
    for client_id, indices in client_indices.items():
        client_data = data[indices]
        client_labels = labels[indices]
        client_ages = ages[indices]
        
        # Use a deterministic random state for each client
        client_random_state = random_state + int(client_id) * 100
        
        X_train, X_val, y_train, y_val, age_train, age_val = train_test_split(
            client_data, client_labels, client_ages,
            test_size=validation_size, 
            random_state=client_random_state,
            stratify=client_labels
        )
        
        # Create balanced dataset for training
        train_dataset = BalancedClientDataset(X_train, y_train, age_train, transform=transform)
        val_dataset = ClientDataset(X_val, y_val, age_val, transform=transform)
        
        # For reproducibility in DataLoader
        g = torch.Generator()
        g.manual_seed(client_random_state)
        
        clients[f'client_{client_id}'] = {
            'train_loader': DataLoader(
                train_dataset, 
                batch_size=batch_size,
                shuffle=True,
                worker_init_fn=lambda _: np.random.seed(client_random_state),
                generator=g
            ),
            'val_loader': DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False
            )
        }
    return clients


'''def evaluate_global_model(X_test, Y_test, sensitive_attr_test, model, sensitive_attribute_name="age"):
    # Get model predictions (probabilities)
    y_pred_probs = model.predict(X_test)

    # Convert probabilities to binary labels (threshold 0.5)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()  # Convert to shape (num_samples,)

    # Ensure true labels are also in the correct format
    y_true = Y_test.flatten()  # Convert to shape (num_samples,)

    # Compute standard classification metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='binary', zero_division=0)
    rec = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

    # Compute fairness metrics (Equalized Odds Difference & Statistical Parity Difference)
    eod, spd = calculate_eod_spd(y_true, y_pred, sensitive_attr_test, sensitive_attribute_name)

    # Display metrics
    print(f"Global Model Test Accuracy: {acc * 100:.2f}%")
    print(f"Global Model Test Precision: {prec * 100:.2f}%")
    print(f"Global Model Test Recall: {rec * 100:.2f}%")
    print(f"Global Model Test F1 Score: {f1 * 100:.2f}%")
    print(f"Equalized Odds Difference: {eod}")
    print(f"Statistical Parity Difference: {spd}")

    return acc, prec, rec, f1, eod, spd'''

import gc
gc.collect()
torch.cuda.empty_cache()

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
gc.collect()
torch.cuda.empty_cache()

def build_efficientnet_model():
    model = models.efficientnet_b0(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(model.classifier[1].in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
        nn.Sigmoid()  # or use BCEWithLogitsLoss and remove this
    )
    return model
    
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for X_batch, y_batch, _ in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1).float()
        #X_batch = X_batch.permute(0, 2, 1, 3)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
        preds = (outputs > 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
    return running_loss / total, correct / total
    
@torch.no_grad()
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_ages = []
    with torch.no_grad():
        for X_batch, y_batch, age_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1).float()
            #X_batch = X_batch.permute(0, 2, 1, 3)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_ages.extend(age_batch.numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels), np.array(all_ages)


def compute_spd(y_true, y_pred, sensitive_attr):
    y_pred = np.array(y_pred).flatten()
    sensitive_attr = np.array(sensitive_attr).flatten()
    return abs(np.mean(y_pred[sensitive_attr == 0]) - np.mean(y_pred[sensitive_attr == 1]))

def compute_eod(y_true, y_pred, sensitive_attr):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    sensitive_attr = np.array(sensitive_attr).flatten()

    mask_pos = y_true == 1
    tpr_0 = np.mean(y_pred[(sensitive_attr == 0) & mask_pos])
    tpr_1 = np.mean(y_pred[(sensitive_attr == 1) & mask_pos])
    return abs(tpr_0 - tpr_1)
    
def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    else:
        return obj

def main():
    # Set up command line argument parsing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description='Run federated learning with DeepSMOTE.')
    parser.add_argument('--alpha', type=float, default=0.3, 
                        help='Alpha parameter for Dirichlet distribution (default: 0.3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--min_samples', type=int, default=80,
                        help='Minimum samples per demographic group per client (default: 80)')
    
    # Parse the arguments
    args_cmd = parser.parse_args()
    alpha = args_cmd.alpha
    seed = args_cmd.seed
    min_samples = args_cmd.min_samples
    
    # Set all random seeds for reproducibility
    seed = args_cmd.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = f"run_case0_noDeepSMOTE_a{alpha}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Model parameters
    args = {
        'dim_h': 32,
        'n_channel': 3,
        'n_z': 64,
        'lr': 0.0002,
        'epochs': 150,
        'batch_size': 16,
        'save': True,
        'train': True,
        'dataset': 'utkface'
    }
    
    print(f"\n===== Experiment Parameters =====")
    print(f"Alpha: {args_cmd.alpha}")
    print(f"Random seed: {seed}")
    print(f"Minimum samples per demographic group: {min_samples}")
    print(f"==============================\n")
    
    # Load and preprocess the base data (this remains constant)
    image_paths = glob.glob("./UTKFace/*.jpg.chip.jpg")
    data, labels, age_bins = load_utkface(image_paths)
    
    data_reshaped = data.reshape(-1, 64, 64, 3)
    print("Total samples before filtering:", len(labels))
    valid_indices = np.where(labels <= 1)
    data_cleaned = data_reshaped[valid_indices]
    print("Samples with gender 0 or 1:", len(valid_indices[0]))
    
    labels_cleaned = labels[valid_indices]
    age_bins = age_bins[valid_indices]
    
    # For reproducibility: Use a consistent random state for all splits
    # derived from the main seed but independent of alpha
    data_split_seed = seed * 10  
    
    # Use numpy's RandomState for consistent splitting
    rng = np.random.RandomState(data_split_seed)
    
    total_samples = len(data_cleaned)
    server_fraction = 0.2  # 20% of training data goes to server

    indices = np.arange(total_samples)
    rng.shuffle(indices)
    server_count = int(server_fraction * total_samples)

    server_indices = indices[:server_count]
    client_indices_all = indices[server_count:]

    X_server = data_cleaned[server_indices]
    Y_server = labels_cleaned[server_indices]
    age_server = age_bins[server_indices]

    X_clients_all = data_cleaned[client_indices_all]
    Y_clients_all = labels_cleaned[client_indices_all]
    age_clients_all = age_bins[client_indices_all]

    X_train, X_test, Y_train, Y_test, age_train, age_test = train_test_split(
    X_clients_all, Y_clients_all, age_clients_all, test_size=0.2,
    random_state=seed, stratify=Y_clients_all)
    
    print("=== Server Data Distribution ===")
    print("Age bins (0=under 50, 1=50+):", dict(Counter(age_server)))
    print("Gender labels (0=male, 1=female):", dict(Counter(Y_server)))
    print("Joint Age-Gender:")
    from collections import defaultdict
    joint = defaultdict(int)
    for a, g in zip(age_server, Y_server):
        joint[(a, g)] += 1
    for key, count in joint.items():
        print(f"  Age={key[0]}, Gender={key[1]}: {count} samples")
    
    # Apply DeepSMOTE on server data
    #X_server_chw = np.transpose(X_server, (0, 3, 1, 2))
    #X_server_smote, Y_server_smote, age_server_smote = apply_deepsmote(X_server_chw, Y_server, age_server, args)
    #X_server_smote = np.transpose(X_server_smote, (0, 2, 3, 1))

    server_train, server_val, Y_train_srv, Y_val_srv, age_train_srv, age_val_srv = train_test_split(
        X_server, Y_server, age_server, test_size=0.2,
        random_state=22, stratify=Y_server)
        
    # Create datasets for server training
    server_train_ds = BalancedClientDataset(server_train, Y_train_srv, age_train_srv)
    server_val_ds = ClientDataset(server_val, Y_val_srv, age_val_srv)

    #server_train_ds = ClientDataset(server_train, Y_train_srv, age_train_srv)
    #server_val_ds = ClientDataset(server_val, Y_val_srv, age_val_srv)

    #server_train_loader = DataLoader(server_train_ds, batch_size=32, shuffle=True)
    # Create data loaders with reproducible settings
    g = torch.Generator()
    g.manual_seed(data_split_seed)
    
    server_train_loader = DataLoader(
        server_train_ds, 
        batch_size=32, 
        shuffle=True,
        worker_init_fn=lambda _: np.random.seed(data_split_seed),
        generator=g
    )
    
    server_val_loader = DataLoader(server_val_ds, batch_size=32, shuffle=False)
    
    test_dataset = ClientDataset(X_test, Y_test, age_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("\n[INFO] Pretraining model on server data...")
    global_model = build_efficientnet_model().to(device)
    optimizer = torch.optim.Adam(global_model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    server_train_losses = []
    server_train_accuracies = []
    server_train_spds = []
    server_train_eods = []

    # Modify the server pretraining loop
    for epoch in range(10):  # epochs of server-side pretraining
        train_loss, train_acc = train_one_epoch(global_model, server_train_loader, optimizer, criterion, device)
        val_loss, val_acc, preds, labels, ages = evaluate_model(global_model, server_val_loader, criterion, device)
        
        # Compute SPD and EOD
        spd = compute_spd(labels, preds, ages)
        eod = compute_eod(labels, preds, ages)
        
        # Store metrics
        server_train_losses.append(train_loss)
        server_train_accuracies.append(train_acc)
        server_train_spds.append(spd)
        server_train_eods.append(eod)
        
        # Print metrics
        print(f"  [Server Pretrain Epoch {epoch+1}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, SPD: {spd:.4f}, EOD: {eod:.4f}")

    # After the server pretraining loop, add plots for SPD and EOD
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), server_train_spds, label="SPD")
    plt.plot(range(1, 11), server_train_eods, label="EOD")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("Server Pretraining Fairness Metrics (SPD & EOD)")
    plt.legend()
    plt.grid(True)
    plt.savefig("ServerPretrainFairnessMetrics_case_0.png")
    plt.close()

    # Create balanced client partitions ensuring minimum samples per demographic group
    '''client_partition = create_balanced_client_partition(
        X_train, Y_train, age_train, 
        num_clients=5, 
        alpha=alpha, 
        seed=seed,
        min_samples_per_group=min_samples
    )
    
    # Assign clients with balanced sampling
    clients = assign_clients_with_balanced_sampling(
        X_train, Y_train, age_train,
        client_partition,
        validation_size=0.2, 
        batch_size=32
    )'''
    
    client_partition = dirichlet_partition(
        X_train, age_train, Y_train, 
        num_clients=5, 
        alpha=alpha, 
        seed=seed,
        min_samples_per_group=min_samples
    )
    
    clients = assign_clients_with_balanced_sampling(
        X_train, Y_train, age_train,
        client_partition,
        validation_size=0.2, 
        batch_size=32,
        random_state=seed
    )
    #np.random.shuffle(clients)
    
    '''clients = assign_clients(
    X_train, Y_train, age_train,
    dirichlet_partition(X_train, age_train, Y_train, num_clients=5, alpha=alpha),
    validation_size=0.2, batch_size=32)
    
    check_client_distribution(clients)'''
    
    # Entropy tracking across rounds
    entropy_per_round = []
    global_metrics_per_round = []
    global_accuracies = []
    global_spds = []
    global_eods = []
    global_losses = []

    for comm_round in range(10):
        print(f"Communication Round {comm_round + 1}/10")
        scaled_weights = []
        round_entropy = {}
        client_fairness = {}
        client_loaders = {}
        for client_name, client in clients.items():
            model = build_efficientnet_model().to(device)
            model.load_state_dict(global_model.state_dict())
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.BCELoss()

            train_loader = client['train_loader']
            val_loader = client['val_loader']
            client_loaders[client_name] = client 

            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, preds, labels, ages = evaluate_model(model, val_loader, criterion, device)

            pred_probs = preds.flatten()
            pred_bin = (pred_probs > 0.5).astype(int)
            p_pos = np.mean(pred_bin)
            round_entropy[client_name] = entropy([p_pos, 1 - p_pos], base=2)

            client_spd = compute_spd(labels, preds, ages)
            client_eod = compute_eod(labels, preds, ages)
            client_fairness[client_name] = {"spd": client_spd, "eod": client_eod}

            print(f"  {client_name} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Entropy: {round_entropy[client_name]:.4f}, SPD: {client_spd:.4f}, EOD: {client_eod:.4f}")

            scalar = len(train_loader.dataset)
            scaled = scale_weights(model.state_dict(), scalar)
            scaled_weights.append(scaled)

        total_samples = sum(len(client['train_loader'].dataset) for client in clients.values())
        avg_weights = sum_weights([scale_weights(w, 1.0 / total_samples) for w in scaled_weights])
        #avg_weights = reweighted_fedavg(scaled_weights, client_loaders, num_classes=2)
        global_model.load_state_dict(avg_weights)
        entropy_per_round.append({"entropy": round_entropy, "client_fairness": client_fairness})

        # Global evaluation after each round
        test_loss, test_acc, preds, labels, ages = evaluate_model(global_model, test_loader, criterion, device)
        spd = compute_spd(labels, preds, ages)
        eod = compute_eod(labels, preds, ages)
        print(f"[Round {comm_round + 1}] Global Eval - Accuracy: {test_acc * 100:.2f}%, Global Model Test Loss: {test_loss:.4f}, SPD: {spd:.4f}, EOD: {eod:.4f}")
        global_accuracies.append(test_acc)
        global_losses.append(test_loss)
        global_spds.append(spd)
        global_eods.append(eod)

    # Final test evaluation
    print("\n[INFO] Final Evaluation of Global Model after All Rounds")
    test_loss, test_acc, preds, labels, ages = evaluate_model(global_model, test_loader, criterion, device)
    spd = compute_spd(labels, preds, ages)
    eod = compute_eod(labels, preds, ages)
    print(f" - Accuracy: {test_acc * 100:.2f}%")
    print(f" - Equalized Odds Difference (EOD): {eod:.4f}")
    print(f" - Statistical Parity Difference (SPD): {spd:.4f}")
    
    # Prepare data
    rounds = list(range(1, len(entropy_per_round) + 1))
    clients_list = list(clients.keys())

    # Initialize dictionaries
    client_entropies = {client: [] for client in clients_list}
    client_spds = {client: [] for client in clients_list}
    client_eods = {client: [] for client in clients_list}

    # Populate metrics
    for round_data in entropy_per_round:
        for client in clients_list:
            client_entropies[client].append(round_data["entropy"][client])
            client_spds[client].append(round_data["client_fairness"][client]["spd"])
            client_eods[client].append(round_data["client_fairness"][client]["eod"])
    
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, global_spds, label="Global SPD")
    plt.plot(rounds, global_eods, label="Global EOD")
    plt.xlabel("Communication Round")
    plt.ylabel("Metric Value")
    plt.title("Global Metrics over Rounds")
    plt.legend()
    plt.grid(True)
    plt.savefig("global_metrics_case_0.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, global_accuracies, label="Global Accuracy")
    plt.xlabel("Communication Round")
    plt.ylabel("Metric Value")
    plt.title("Global Accuracy over Rounds")
    plt.legend()
    plt.grid(True)
    plt.savefig("global_accuracy_case_0.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, global_losses, label="Global Loss")
    plt.xlabel("Communication Round")
    plt.ylabel("Metric Value")
    plt.title("Global Loss over Rounds")
    plt.legend()
    plt.grid(True)
    plt.savefig("global_loss_case_0.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    for client in clients_list:
        plt.plot(rounds, client_entropies[client], label=client)
    plt.xlabel("Communication Round")
    plt.ylabel("Entropy")
    plt.title("Client Prediction Entropy over Rounds")
    plt.legend()
    plt.grid(True)
    plt.savefig("client_entropy_case_0.png")  # Save figure
    plt.close()  # Close the figure to avoid overlap
    
    plt.figure(figsize=(10, 6))
    for client in clients_list:
        plt.plot(rounds, client_spds[client], label=client)
    plt.xlabel("Communication Round")
    plt.ylabel("Statistical Parity Difference (SPD)")
    plt.title("Client SPD over Rounds")
    plt.legend()
    plt.grid(True)
    plt.savefig("client_spd_case_0.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    for client in clients_list:
        plt.plot(rounds, client_eods[client], label=client)
    plt.xlabel("Communication Round")
    plt.ylabel("Equalized Odds Difference (EOD)")
    plt.title("Client EOD over Rounds")
    plt.legend()
    plt.grid(True)
    plt.savefig("client_eod_case_0.png")
    plt.close()
    
    plt.figure(figsize=(10,4))
    plt.subplot(1, 2, 1)
    plt.plot(server_train_losses, label='Loss')
    plt.title("Server Pretraining Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("ServerPretrainLoss_case_0.png")
    plt.close()

    plt.subplot(1, 2, 2)
    plt.plot(server_train_accuracies, label='Accuracy', color='green')
    plt.title("Server Pretraining Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("ServerPretrainAccuracy_case_0.png")
    plt.close()


if __name__ == "__main__":
    main()