import yaml
import torch
from model import RawNet2
import os
import torchaudio
from dataset import pad_with_copy
import torch.nn.functional as F


if __name__ == "__main__":
    SEED = 123
    torch.manual_seed(SEED)

    config_path = '/home/ubuntu/as_project/configs/config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = RawNet2(args=config['model']).to(device)

    load_path = config['train']['load_path']

    model_dict = torch.load(load_path)
    model.load_state_dict(model_dict['model'])
    print("model loaded")

    for audio_path in os.listdir(config['dataset']['custom_test']['dataset_path']):
        audio, sr = torchaudio.load(f"{config['dataset']['custom_test']['dataset_path']}/{audio_path}")
        audio = audio[0]
        audio = pad_with_copy(audio=audio, max_len=config['dataset']['train']['max_len']).view(1, 1, -1)
        audio = audio.to(device)
        pred = model(audio)
        pred_proba = F.softmax(pred, dim=-1)

        print(audio_path, pred_proba[0][0].item())
