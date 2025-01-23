import os
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import librosa
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import csv
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import precision_recall_fscore_support
import logging
import time
from tqdm import tqdm

logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_features(audio_file, sr=24000, n_mels=128, n_fft=1024, hop_length=256):
    y, sr = librosa.load(audio_file, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def load_annotations(annotation_file):
    annotations = []
    with open(annotation_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            onset, offset = float(row[0]), float(row[1])
            annotations.append((onset, offset))
    return annotations


def extract_and_save_features_and_labels(audio_dir, annotations_dir, feature_file, label_file, sr=24000, n_mels=128, n_fft=1024, hop_length=256, compression='gzip', compression_opts=9):
    if not os.path.exists(feature_file) or not os.path.exists(label_file):
        with h5py.File(feature_file, 'w') as hf_features, h5py.File(label_file, 'w') as hf_labels:
            audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
            annotation_files = sorted([f for f in os.listdir(annotations_dir) if f.endswith('.csv')])

            for audio_file, annotation_file in zip(audio_files, annotation_files):
                audio_path = os.path.join(audio_dir, audio_file)
                annotation_path = os.path.join(annotations_dir, annotation_file)


                mel_spec_db = extract_features(audio_path, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
                mel_spec_db = (mel_spec_db - np.mean(mel_spec_db)) / np.std(mel_spec_db)  # 归一化
                hf_features.create_dataset(audio_file, data=mel_spec_db, compression=compression, compression_opts=compression_opts)


                annotations = load_annotations(annotation_path)
                num_frames = mel_spec_db.shape[1]
                labels = np.zeros(num_frames, dtype=int)
                for onset, offset in annotations:
                    start_frame = int(onset * sr / hop_length)
                    end_frame = int(offset * sr / hop_length)
                    start_frame = max(start_frame, 0)
                    end_frame = min(end_frame, num_frames)
                    if start_frame < num_frames and end_frame > 0:
                        labels[start_frame:end_frame] = 1
                hf_labels.create_dataset(audio_file, data=labels, compression=compression, compression_opts=compression_opts)
    else:
        print(f"Feature file {feature_file} and/or label file {label_file} already exists. Skipping extraction.")



extract_and_save_features_and_labels('output3/train/wavefile', 'output3/train/annotation', 'output3/train/features.h5', 'output3/train/labels.h5')
extract_and_save_features_and_labels('output3/validation/wavefile', 'output3/validation/annotation', 'output3/validation/features.h5', 'output3/validation/labels.h5')
extract_and_save_features_and_labels('output3/test/wavefile', 'output3/test/annotation', 'output3/test/features.h5', 'output3/test/labels.h5')
