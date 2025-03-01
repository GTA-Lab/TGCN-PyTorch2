# T-GCN-PyTorch2


This is a PyTorch2 implementation of T-GCN in the following paper: [T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction](https://arxiv.org/abs/1811.05320).
Here’s a complete and polished `ReadMe` for the repository that has been updated from PyTorch Lightning to PyTorch 2.x:

---

# T-GCN Implementation in PyTorch 2.x

This repository provides a **PyTorch 2.x** implementation of the **Temporal Graph Convolutional Network (T-GCN)** model, originally proposed in the paper:  
**[T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction](https://arxiv.org/abs/1811.05320)**.  

The original implementation was based on **PyTorch Lightning** and can be found here:  
[https://github.com/martinwhl/T-GCN-PyTorch](https://github.com/martinwhl/T-GCN-PyTorch).  

This repository contains an updated version of the code, migrated to **PyTorch 2.x**, with improvements for better compatibility, performance, and ease of use.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Dataset](#dataset)
4. [Usage](#usage)
5. [Contributors](#contributors)
6. [Citation](#citation)
7. [License](#license)

---

## Introduction
T-GCN is a deep learning model designed for traffic prediction, combining **Graph Convolutional Networks (GCN)** and **Gated Recurrent Units (GRU)** to capture both spatial and temporal dependencies in traffic data. This implementation provides a PyTorch 2.x version of the original model, leveraging the latest features and optimizations in PyTorch.

---

## Requirements
To run the code, ensure you have the following dependencies installed:
- Python 3.x
- PyTorch 2.x
- NumPy
- Pandas
- Torchmetrics (>= 0.11)

You can install the required packages using:
```bash
pip install -r requirements.txt
```

---

## Dataset
The model is trained and evaluated on traffic datasets, which typically include traffic speed or flow data. The dataset used in the original paper are included in the `data/` folder.

---

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/GTA-Lab/TGCN-PyTorch2.git
   cd TGCN-PyTorch2
   ```
2. Use CLI with YAML config file such as:

    ```bash
    # T-GCN
    python  main.py --config configs/tgcn.yaml --log_path train.log 
    ```

or run main.ipynb

---

## Contributors
- **Amiri & Mostafazade**: Migrated the original PyTorch Lightning code to PyTorch 2.x and improved the codebase.  
- **Original Authors**: The T-GCN model was originally developed by the authors of the paper linked above.

---

## Citation
If you use this code or the T-GCN model in your research, please cite the original paper:
```bibtex
@article{doi:10.1109/TITS.2019.2935152,
  author = {Zhao, Ling and Song, Yu and Zhang, Chao and Liu, Yu and Wang, Pu and Lin, Tao and Deng, Min and Li, Haifeng},
  title = {T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  volume = {21},
  number = {9},
  pages = {3848-3858},
  year = {2020},
  doi = {10.1109/TITS.2019.2935152}
}
```

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
