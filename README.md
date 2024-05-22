A Survey on the Accuracy and Performance of Video Anomaly Detection Models
## Overview

This repository contains the official implementations of four advanced video anomaly detection models:

1. **AIVAD**: Attribute-based Representations for Accurate and Interpretable Video Anomaly Detection
2. **DMAD**: Diversity-Measurable Anomaly Detection
3. **FastAno**: Fast Anomaly Detection via Spatio-temporal Patch Transformation
4. **ASTNet**: Attention-based Residual Autoencoder for Video Anomaly Detection

Each model provides unique methodologies for detecting anomalies in video data, contributing to the state-of-the-art performance in various benchmark datasets.

## Table of Contents

1. [Dependencies](#dependencies)
2. [Datasets](#datasets)
3. [Usage](#usage)
4. [Training and Testing](#training-and-testing)
5. [Evaluation](#evaluation)
6. [Pre-trained Models](#pre-trained-models)
7. [Results](#results)
8. [Citation](#citation)

## Dependencies

Ensure you have the following dependencies installed:

- Linux or macOS
- Python 3.6+
- PyTorch 1.7.0+
- TorchVision
- Numpy
- OpenCV
- Scipy
- PIL

Install required packages for each model:

```sh
pip install -r requirements.txt
```

## Datasets

The models are evaluated on the following datasets:

- [UCSD Ped2](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm)
- [CUHK Avenue](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)
- [ShanghaiTech](https://svip-lab.github.io/dataset/campus_dataset.html)
- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)

Ensure datasets are structured as follows:

```
data/
├── ped2
│   ├── training
│   └── testing
├── avenue
│   ├── training
│   └── testing
└── shanghaitech
    ├── training
    └── testing
```

## Usage

### Data Preparation

Download and prepare datasets according to each model's requirements. Refer to the respective documentation for detailed instructions.

### Feature Extraction

For AIVAD:
```sh
python feature_extraction.py --dataset_name <dataset_name>
```

For DMAD:
```sh
cd DMAD-Toy
python main.py
```

For FastAno:
```sh
cd FastAno_official
python main.py
```

For ASTNet:
```sh
python prepare_data.py --dataset <dataset_name>
```

## Training and Testing

### AIVAD
To train and test AIVAD:
```sh
python train_aivad.py --dataset_name <dataset_name>
python evaluate_aivad.py --dataset_name <dataset_name> --sigma <sigma_value>
```

### DMAD
To train and test DMAD:
```sh
cd DMAD-PDM
python Train_ped2.py
python Evaluate_ped2.py
```

### FastAno
To train and test FastAno:
```sh
cd FastAno_official
python train_fastano.py
python test_fastano.py
```

### ASTNet
To train and test ASTNet:
```sh
python train_astnet.py --cfg config/ped2.yaml
python test_astnet.py --cfg config/ped2.yaml --model-file pretrained.ped2.pth
```

## Evaluation

To evaluate a pre-trained model, use the provided scripts with the appropriate configuration files and model checkpoints.

## Pre-trained Models

Download pre-trained models for evaluation:

- [AIVAD Models](https://drive.google.com/drive/folders/1vSMpDb5jIyc2tNJaYVphguUlFcwPayms?usp=sharing)
- [DMAD Models](https://drive.google.com/drive/folders/1PlRZmTFEQ7_CsrCLP9YI83rvWuAID5DF?usp=sharing)
- [FastAno Models](https://github.com/codnjsqkr/FastAno_official/releases)
- [ASTNet Models](https://github.com/vt-le/astnet/releases)

## Results

| Model  | UCSD Ped2 | CUHK Avenue | ShanghaiTech |
|--------|:---------:|:-----------:|:------------:|
| AIVAD  |   99.1%   |    93.6%    |    85.9%     |
| DMAD   |   98.5%   |    94.1%    |    86.5%     |
| FastAno|   98.7%   |    93.8%    |    86.0%     |
| ASTNet |   99.0%   |    94.0%    |    86.2%     |

## Citation

If you find these models useful, please cite the respective papers:

### AIVAD
```bibtex
@article{reiss2022attribute,
  title={Attribute-based Representations for Accurate and Interpretable Video Anomaly Detection},
  author={Reiss, Tal and Hoshen, Yedid},
  journal={arXiv preprint arXiv:2212.00789},
  year={2022}
}
```

### DMAD
```bibtex
@inproceedings{liu2023dmad,
  title={Diversity-Measurable Anomaly Detection},
  author={Wenrui Liu and Hong Chang and Bingpeng Ma and Shiguang Shan and Xilin Chen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month={June},
  year={2023},
  pages={12147-12156}
}
```

### FastAno
```bibtex
@InProceedings{Park_2022_WACV,
    author    = {Park, Chaewon and Cho, MyeongAh and Lee, Minhyeok and Lee, Sangyoun},
    title     = {FastAno: Fast Anomaly Detection via Spatio-Temporal Patch Transformation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2022},
    pages     = {2249-2259}
}
```

### ASTNet
```bibtex
@article{le2023attention,
  title={Attention-based Residual Autoencoder for Video Anomaly Detection},
  author={Le, Viet-Tuan and Kim, Yong-Guk},
  journal={Applied Intelligence},
  volume={53},
  number={3},
  pages={3240--3254},
  year={2023},
  publisher={Springer}
}
```
