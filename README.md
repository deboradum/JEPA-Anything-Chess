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
