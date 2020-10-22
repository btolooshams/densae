[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# DenSaE (dense and sparse autoencoder)

### Trained Models

Trained models are stored in `results/trained_models`.

##
### PATH

For any scripts to run, make sure you are in `src` directory.

##
### Configuration


Create a configuration function in `conf.py` containing a dictionary of hyperparameters for your experiment.

```
@config_ingredient.named_config
def exp1():
    hyp = {
        "experiment_name": "noise15_densae_1A_63B_hyp",
        "network": "CSCNetTiedLS",
        "noiseSTD": 15,
        "dictionary_dim": 7,
        "stride": 5,
        "strideA": 5,
        "strideB": 5,
        "split_stride": 5,
        "num_conv_A": 1,
        "num_conv_B": 63,
        "L": 10,
        "num_iters": 15,
        "twosided": True,
        "batch_size": 1,
        "num_epochs": 250,
        "normalize": False,
        "lr": 1e-4,
        "lr_decay": 0.80,
        "lr_step": 50,
        "info_period": 10000,
        "model_period": 10000,
        "loss_period": 10000,
        "crop_dim": (128, 128),
        "lam": 0.085,
        "rho": 1e10,
        "weight_decay": 0,
        "supervised": True,
        "shuffle": True,
        "denoising": True,
        "loss": "l2",
        "train_path": "../data/CBSD432/",
        "test_path": "../data/BSD68/",
    }
```

##
### Training

`python train.py with cfg.exp1`

##
### Results

When training is done, the results are saved in `results/{experiment_name}/{random_date}`.

`random_date` is a datetime string generated at the begining of the training.

##
### Prediction

Run `predict.py`. Make sure to specify the parameters from line 37 - 42.
