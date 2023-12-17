# Anti-spoofing

Based on [RawNet2](https://arxiv.org/pdf/2011.01108.pdf) implementation


[WandB Report](https://wandb.ai/yuliazhelt/as_project/reports/Anti-spoofing--Vmlldzo2Mjc5NTk4?accessToken=qq0pibuqtuivmova1w71xzsq7yzf3jgmnnqmf21f4y06b3wfsxwlx96eojbn1v4n)


## Installation guide

Load trained model:
```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -r ./requirements.txt
python load_model.py
```

## Reproduction
Train for 25 epochs 
```
python main.py
```