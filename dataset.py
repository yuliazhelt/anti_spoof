from torch.utils.data import Dataset
from pathlib import Path
import json
import torchaudio
import torch
import torch.nn.functional as F
from typing import List 
from tqdm.auto import tqdm
import numpy as np
import random
import pandas as pd

    
def pad_with_copy(audio, max_len):
    if len(audio) >= max_len:
        return audio[:max_len]
    num_repeats = (max_len + len(audio) - 1) // len(audio)
    tiled_audio = audio.repeat(num_repeats)
    return tiled_audio[:max_len]


class CustomDataset(Dataset):
    def __init__(self, dataset_path = '.', part='train', max_len=64_000, limit=None):
        super().__init__()
        self.part = part
        if part == 'train':
            LA_PROTOCOL = f'{dataset_path}/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'

        else:
            LA_PROTOCOL = f'{dataset_path}/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{part}.trl.txt'
            
        LA_DIR = f'{dataset_path}/LA/LA/ASVspoof2019_LA_{part}/flac/'
        
        dataset_df = pd.read_csv(LA_PROTOCOL, delimiter=' ',header=None).sample(frac=1,random_state=13).reset_index()
        
        if limit is not None:
            dataset_df = dataset_df[:limit]
    
        self.audio_paths = np.array([LA_DIR+path+'.flac' for path in dataset_df[1]])

        dataset_labels = dataset_df[4].to_numpy()
        dataset_labels_int = []
        
        for label in dataset_labels:
            dataset_labels_int.append(0 if label == 'spoof' else 1)  
            
        self.dataset_labels = dataset_labels_int
        self.max_len = max_len

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, ind):
        audio_path = self.audio_paths[ind]
        audio, sr = torchaudio.load(audio_path)
        audio = audio[0]
        # padding with copy
        audio = pad_with_copy(audio=audio, max_len=self.max_len)

        label = self.dataset_labels[ind]
        return {
            "audio_path": audio_path,
            "label": label,
            "audio": audio,
            "sample_rate": sr
        }


def collate_fn(dataset_items: List[dict]):
    audios = []
    labels = []
    for elem in dataset_items:
        audios.append(elem["audio"])
        labels.append(elem["label"])
    return torch.stack(audios).unsqueeze(1), torch.tensor(labels)