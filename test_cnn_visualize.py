import torch
import torch.nn as nn
import numpy as np
import librosa
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from sklearn.metrics import precision_score, recall_score, f1_score


class CNN(nn.Module):
    def __init__(self, input_height, input_width, num_classes, conv_channels=[32, 64, 128], fc_size=256):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, conv_channels[0], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_channels[0])
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_channels[1])
        self.conv3 = nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_channels[2])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.1)

        conv_output_height = input_height // 8
        conv_output_width = input_width // 8
        fc_input_size = conv_channels[2] * conv_output_height * conv_output_width

        self.fc1 = nn.Linear(fc_input_size, fc_size)
        self.fc2 = nn.Linear(fc_size, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
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


def extract_features(audio_file, sr=24000, n_mels=128, n_fft=1024, hop_length=256):
    y, sr = librosa.load(audio_file, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db, y, sr

def load_annotations(annotation_file):
    annotations = []
    with open(annotation_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            onset, offset = float(row[0]), float(row[1])
            annotations.append((onset, offset))
    return annotations

def segment_windows(spectrogram, window_size=16, hop_size=16):
    windows = []
    num_windows = (spectrogram.shape[1] - window_size) // hop_size + 1
    for i in range(num_windows):
        start = i * hop_size
        end = start + window_size
        window = spectrogram[:, start:end]
        windows.append(window)
    return np.array(windows)

def predict_and_visualize(model, audio_file, annotation_file, window_size=16, hop_size=16, threshold=0.5, sr=24000, n_mels=128, n_fft=1024, hop_length=256):
    model.eval()
    device = next(model.parameters()).device

    mel_spec_db, y, sr = extract_features(audio_file, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_spec_db = (mel_spec_db - np.mean(mel_spec_db)) / np.std(mel_spec_db)
    windows = segment_windows(mel_spec_db, window_size, hop_size)
    windows = torch.tensor(windows).unsqueeze(1).float().to(device)

    with torch.no_grad():
        outputs = model(windows)
    predictions = outputs.cpu().numpy()

    num_classes = predictions.shape[1]
    aggregated_predictions = np.zeros((predictions.shape[0] * hop_size + window_size, num_classes))
    for i, window_pred in enumerate(predictions):
        start = i * hop_size
        end = start + window_size
        aggregated_predictions[start:end] += window_pred

    # Find onsets and offsets for each class
    onsets = [[] for _ in range(num_classes)]
    offsets = [[] for _ in range(num_classes)]
    for class_idx in range(num_classes):
        in_event = False
        for i, value in enumerate(aggregated_predictions[:, class_idx]):
            if value > threshold and not in_event:
                onsets[class_idx].append(i)
                in_event = True
            elif value <= threshold and in_event:
                offsets[class_idx].append(i)
                in_event = False
        if in_event:
            offsets[class_idx].append(len(aggregated_predictions))

    # Convert frame indices to time
    for class_idx in range(num_classes):
        onsets[class_idx] = np.array(onsets[class_idx]) * hop_length / sr
        offsets[class_idx] = np.array(offsets[class_idx]) * hop_length / sr

    base_path = 'output3/test/predictedannotation/'
    os.makedirs(base_path, exist_ok=True)
    audio_file_name = os.path.basename(audio_file)
    file_prefix = audio_file_name.split('.')[0].replace('play_', '')  # Remove 'play_' prefix if present
    predicted_annotation_file = os.path.join(base_path, f'annotation_{file_prefix}_predicted.csv')

    with open(predicted_annotation_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for class_idx in range(num_classes):
            for onset, offset in zip(onsets[class_idx], offsets[class_idx]):
                writer.writerow([onset, offset, f'Class_{class_idx}'])

    print(f"Predicted annotation saved to: {predicted_annotation_file}")

    annotations = load_annotations(annotation_file)
    ground_truth_onsets = [ann[0] for ann in annotations]
    ground_truth_offsets = [ann[1] for ann in annotations]

    # Plotting
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.plot(np.linspace(0, len(y) / sr, len(y)), y, label='Waveform')
    light_red = (1, 0.5, 0.5, 0.3)
    label_added = False

    for class_idx in range(num_classes):
        for onset, offset in zip(onsets[class_idx], offsets[class_idx]):
            if not label_added:
                plt.axvspan(onset, offset, color=light_red, alpha=0.3, label='Detected Event')
                label_added = True
            else:
                plt.axvspan(onset, offset, color=light_red, alpha=0.3)

    plt.title('Model Predictions')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(np.linspace(0, len(y) / sr, len(y)), y, label='Waveform')
    for onset, offset in zip(ground_truth_onsets, ground_truth_offsets):
        plt.axvspan(onset, offset, color='b', alpha=0.3, label='Ground Truth Event' if onset == ground_truth_onsets[0] else "")
    plt.title('Ground Truth')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    input_height = 128
    input_width = 16
    num_classes = 16

    model = CNN(input_height=input_height, input_width=input_width, num_classes=num_classes).to(device)
    criterion = FocalLoss()

    checkpoint_path = 'model_cnn.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {checkpoint_path}")

    sr = 24000
    n_mels = 128
    n_fft = 1024
    hop_length = 256
    window_size = 16
    hop_size = 16


    audio_file = 'output3/test/wavefile/play_309.wav'
    annotation_file = 'output3/test/annotation/annotation_309.csv'
    predict_and_visualize(model, audio_file, annotation_file, window_size, hop_size, threshold=0.5, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
