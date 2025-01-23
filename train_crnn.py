import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from sklearn.metrics import precision_score, f1_score
import logging
import time
from tqdm import tqdm
import gc


logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CRNN(nn.Module):
    def __init__(self, input_height, input_width, num_classes, conv_channels=[16, 32], lstm_hidden_size=64,
                 num_layers=2):
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv2d(1, conv_channels[0], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        conv_output_height = input_height // 4
        conv_output_width = input_width // 4
        conv_output_channels = conv_channels[1]
        lstm_input_size = conv_output_channels * conv_output_width

        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size(0), x.size(1), -1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        x = self.sigmoid(x)
        return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(F_loss)

def find_optimal_threshold(outputs, labels):
    thresholds = np.linspace(0, 1, 100)
    f1_scores = [f1_score(labels, (outputs > t).astype(int), average='weighted', zero_division=1) for t in thresholds]
    return thresholds[np.argmax(f1_scores)]

class OptimizedDataset(Dataset):
    def __init__(self, feature_file, label_file, window_size=16, hop_size=16, cache_size=1000, augment=False, augment_prob_ratio=5):
        self.feature_file = h5py.File(feature_file, 'r')
        self.label_file = h5py.File(label_file, 'r')
        self.window_size = window_size
        self.hop_size = hop_size
        self.cache_size = cache_size
        self.cache = {}
        self.augment = augment
        self.augment_prob_ratio = augment_prob_ratio

        self.feature_keys = list(self.feature_file.keys())
        self.total_windows = sum(self.compute_num_windows(self.feature_file[key].shape[1]) for key in self.feature_keys)

    def compute_num_windows(self, num_frames):
        return (num_frames - self.window_size) // self.hop_size + 1

    def __len__(self):
        return self.total_windows

    def add_noise(self, mel_spec):
        noise_level = np.random.uniform(0.0001, 0.005)
        noise = np.random.normal(0, noise_level, mel_spec.shape)
        return mel_spec + noise

    def pitch_shift_mel(self, mel_spec, n_steps=0):
        if n_steps == 0:
            return mel_spec
        shifted = np.zeros_like(mel_spec)
        if n_steps > 0:
            shifted[n_steps:, :] = mel_spec[:-n_steps, :]
        else:
            shifted[:n_steps, :] = mel_spec[-n_steps:, :]
        return shifted

    def frequency_mask(self, mel_spec, num_masks=1, max_width=8):
        for _ in range(num_masks):
            freq = np.random.randint(0, mel_spec.shape[0] - max_width)
            width = np.random.randint(1, max_width)
            mel_spec[freq:freq+width, :] = 0
        return mel_spec

    def time_mask(self, mel_spec, num_masks=1, max_width=8):
        for _ in range(num_masks):
            time = np.random.randint(0, mel_spec.shape[1] - max_width)
            width = np.random.randint(1, max_width)
            mel_spec[:, time:time+width] = 0
        return mel_spec

    def apply_augmentation(self, mel_spec, label):
        zero_class = np.all(label ==0)
        aug_prob = 0.6 if zero_class else 0.4

        if np.random.rand() < aug_prob:
            mel_spec = self.add_noise(mel_spec)
        if np.random.rand() < aug_prob:
            mel_spec = self.pitch_shift_mel(mel_spec, n_steps=np.random.randint(-3, 4))
        if np.random.rand() < aug_prob:
            mel_spec = self.frequency_mask(mel_spec, num_masks=2, max_width=10)
        if np.random.rand() < aug_prob:
            mel_spec = self.time_mask(mel_spec, num_masks=2, max_width=10)
        return mel_spec

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        file_idx, window_idx = self.get_file_and_window_idx(idx)

        feature = self.feature_file[self.feature_keys[file_idx]]
        label = self.label_file[self.feature_keys[file_idx]]

        start_idx = window_idx * self.hop_size
        end_idx = start_idx + self.window_size

        feature_window = feature[:, start_idx:end_idx]
        label_window = label[start_idx:end_idx]

        if self.augment:
            feature_window = self.apply_augmentation(feature_window, label_window)

        item = (torch.tensor(feature_window, dtype=torch.float32),
                torch.tensor(label_window, dtype=torch.float32))

        if len(self.cache) < self.cache_size:
            self.cache[idx] = item

        return item

    def get_file_and_window_idx(self, idx):
        cumulative_windows = 0
        for file_idx, key in enumerate(self.feature_keys):
            num_windows = self.compute_num_windows(self.feature_file[key].shape[1])
            if cumulative_windows + num_windows > idx:
                return file_idx, idx - cumulative_windows
            cumulative_windows += num_windows
        raise IndexError("Index out of range")

    def __del__(self):
        self.feature_file.close()
        self.label_file.close()



def create_data_loader(dataset, batch_size, shuffle, num_workers, prefetch_factor=2):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True
    )



def calculate_metrics(labels, preds, num_classes):
    labels = np.array(labels).reshape(-1, num_classes)
    preds = np.array(preds).reshape(-1, num_classes)
    precision = precision_score(labels, preds, average='weighted', zero_division=1)
    f1 = f1_score(labels, preds, average='weighted', zero_division=1)
    return precision, f1


def log_results(epoch, train_loss, train_precision, train_f1, val_loss, val_precision, val_f1):
    message = (f'Epoch {epoch + 1}, '
               f'Loss: {train_loss:.4f}, Precision: {train_precision:.4f}, F1-score: {train_f1:.4f}, '
               f'Validation Loss: {val_loss:.4f}, Validation Precision: {val_precision:.4f}, Validation F1-score: {val_f1:.4f}')
    print(message)
    logging.info(message)


def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, checkpoint_path, num_classes):
    start_epoch, best_val_f1 = 0, 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_f1 = checkpoint['best_val_f1']
        print(f"Resuming training from epoch {start_epoch}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit='batch'):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()
            with torch.no_grad():
                predicted = (outputs > 0.5).float()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            del inputs, labels, outputs, loss, predicted
            torch.cuda.empty_cache()

        train_loss = running_loss / len(train_loader)
        train_precision, train_f1 = calculate_metrics(all_labels, all_preds, num_classes)

        val_loss, val_precision, val_f1 = validate(model, val_loader, criterion, device, num_classes)

        log_results(epoch, train_loss, train_precision, train_f1, val_loss, val_precision, val_f1)

        scheduler.step(val_loss)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1,
            }, checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")

        gc.collect()
        torch.cuda.empty_cache()


def validate(model, val_loader, criterion, device, num_classes):
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels.float()).item()
            predicted = (outputs > 0.5).float()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            del inputs, labels, outputs, predicted
            torch.cuda.empty_cache()

    val_loss /= len(val_loader)
    val_precision, val_f1 = calculate_metrics(all_labels, all_preds, num_classes)
    return val_loss, val_precision, val_f1


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    input_height = 128
    input_width = 16
    num_classes = 16
    batch_size = 128
    num_workers = 16
    prefetch_factor = 8

    model = CRNN(input_height=input_height, input_width=input_width, num_classes=num_classes).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    train_dataset = OptimizedDataset('output3/train/features.h5', 'output3/train/labels.h5', augment=False)
    val_dataset = OptimizedDataset('output3/validation/features.h5', 'output3/validation/labels.h5', augment=False)
    test_dataset = OptimizedDataset('output3/test/features.h5', 'output3/test/labels.h5', augment=False)

    train_loader = create_data_loader(train_dataset, batch_size, shuffle=True, num_workers=num_workers,
                                      prefetch_factor=prefetch_factor)
    val_loader = create_data_loader(val_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                    prefetch_factor=prefetch_factor)
    test_loader = create_data_loader(test_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                     prefetch_factor=prefetch_factor)

    checkpoint_path = 'model_crnn.pth'
    num_epochs = 35
    train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, checkpoint_path, num_classes)

    test_loss, test_precision, test_f1 = validate(model, test_loader, criterion, device, num_classes)
    print(f'Test Loss: {test_loss:.4f}, Test Precision: {test_precision:.4f}, Test F1-score: {test_f1:.4f}')
    logging.info(f'Test Loss: {test_loss:.4f}, Test Precision: {test_precision:.4f}, Test F1-score: {test_f1:.4f}')
