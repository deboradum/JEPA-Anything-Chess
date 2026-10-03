# JEPA-Anything-Chess

## Environment setup
```
conda create --name jepa-chess python=3.10 -y
conda activate jepa-chess
pip install -r requirements.txt
make install-dev PYTHON=python
make check PYTHON=python
```

## Run training
```
python train.py --config conf.yaml
```

## Status
Failed.

JEPA-anyhting is (obviously) not an appropriate framework for behavioral cloning in Chess.
I thought there was a chance for it to work, but the latent space encoding collapses immediately. My geuss is that this is due to the fact that for discrete spaces like chess games, it is just not practical to predict the next continuous latent space, rather than regular policy prediction.

It was still a fun experiment, though.
