# T-GCN-PyTorch2


This is a PyTorch2 implementation of T-GCN in the following paper: [T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction](https://arxiv.org/abs/1811.05320).

Note that [the original implementation](https://img.shields.io/github/stars/martinwhl/T-GCN-PyTorch) was based on PyTorch lightning.

## Requirements

* numpy
* pandas
* torch
* torchmetrics>=0.11


## Model Training

* CLI

    ```bash
    # GCN
    python main.py fit --trainer.max_epochs 3000 --trainer.accelerator cuda --trainer.devices 1 --data.dataset_name losloop --data.batch_size 64 --data.seq_len 12 --data.pre_len 3 --model.model.class_path models.GCN --model.learning_rate 0.001 --model.weight_decay 0 --model.loss mse --model.model.init_args.hidden_dim 100
    # GRU
    python main.py fit --trainer.max_epochs 3000 --trainer.accelerator cuda --trainer.devices 1 --data.dataset_name losloop --data.batch_size 64 --data.seq_len 12 --data.pre_len 3 --model.model.class_path models.GRU --model.learning_rate 0.001 --model.weight_decay 1.5e-3 --model.loss mse --model.model.init_args.hidden_dim 100
    # T-GCN
    python main.py fit --trainer.max_epochs 1500 --trainer.accelerator cuda --trainer.devices 1 --data.dataset_name losloop --data.batch_size 32 --data.seq_len 12 --data.pre_len 3 --model.model.class_path models.TGCN --model.learning_rate 0.001 --model.weight_decay 0 --model.loss mse_with_regularizer --model.model.init_args.hidden_dim 64
    ```

* YAML config file

    ```bash
    # GCN
    python main.py fit --config configs/gcn.yaml
    # GRU
    python main.py fit --config configs/gru.yaml
    # T-GCN
    python main.py fit --config configs/tgcn.yaml
    ```

Please refer to `python main.py fit -h` for more CLI arguments.

Run `tensorboard --logdir ./lightning_logs` to monitor the training progress and view the prediction results.
