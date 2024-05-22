# A Survey on the Accuracy and Performance of Video Anomaly Detection Models

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

#### AIVAD

**Pre-processing Stage**

To properly perform the pre-processing stage, please change your directory to:

```sh
cd path-to-directory/Interpretable_VAD/
```

1. **Object Detection**

   Our object detector outputs are provided [here](https://drive.google.com/drive/folders/1BnjzuwxyXio2sNU_4w7rlTw4PcURlq_R?usp=sharing).
   Set up the bounding boxes by placing the corresponding files in the following folders:
   - All files for Ped2 should be placed in: `./data/ped2`
   - All files for Avenue should be placed in: `./data/avenue`
   - All files for ShanghaiTech should be placed in: `./data/shanghaitech`

   Install the Detectron2 library:

   ```sh
   python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
   ```

   Download the ResNet50-FPN weights:

   ```sh
   wget https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl -P pre_processing/checkpoints/
   ```

   Detect all the foreground objects:

   ```sh
   python pre_processing/bboxes.py --dataset_name=<dataset_name> [--train]
   ```

   Example for Ped2 training objects:

   ```sh
   python pre_processing/bboxes.py --dataset_name=ped2 --train
   ```

2. **Optical Flow**

   Install FlowNet2.0:

   ```sh
   cd pre_processing
   bash install_flownet2.sh
   cd ..
   ```

   Download pre-trained FlowNet2 weights from [here](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view?usp=sharing) and place it in `Interpretable_VAD/pre_processing/checkpoints/`.

   Estimate all the optical flows:

   ```sh
   python pre_processing/flows.py --dataset_name=<dataset_name> [--train]
   ```

   Example for Ped2 training flows:

   ```sh
   python pre_processing/flows.py --dataset_name=ped2 --train
   ```

**Feature Extraction**

Download the required `pose.npy` file from [here](https://drive.google.com/file/d/1fxMmmZ8TmmdGOovbC2QYPgWTaRxyVdE0/view?usp=sharing) and place it in the following path: `./data/shanghaitech/train/pose.npy`.

Extract features:

```sh
python feature_extraction.py --dataset_name=<dataset_name>
```

**Score Calibration**

Compute calibration parameters:

```sh
python score_calibration.py --dataset_name=<dataset_name>
```

**Evaluation**

Evaluate the model:

```sh
python evaluate.py --dataset_name=<dataset_name> --sigma=<sigma_value>
```

### DMAD

**Toy Experiment**

Download VQ-CVAE-based DMAD-PDM from [GoogleDrive](https://drive.google.com/file/d/1llmszdgp7VvKre-SQDw5GJN3TKs4-IJK/view?usp=sharing) or design your custom dataset:

```sh
cd DMAD-Toy
python main.py
```

**Training & Testing**

Download pre-processing files from [BaiduPan](https://pan.baidu.com/s/1n9ko5szFRjdYxHGbBK0TUw) (Password: dmad) or [GoogleDrive](https://drive.google.com/drive/folders/1PlRZmTFEQ7_CsrCLP9YI83rvWuAID5DF?usp=sharing).

Train or test the PDM version DMAD framework:

```sh
cd DMAD-PDM
python Train_<dataset_name>.py
python Evaluate_<dataset_name>.py
```

Train or test the PPDM version DMAD framework:

```sh
cd DMAD-PPDM
python Train_mvtec.py
python Evaluate_mvtec.py
```

### FastAno

**Setup**

```sh
git clone https://github.com/codnjsqkr/FastAno_official.git
conda create -n fastano python=3.7.9
conda activate fastano
pip install -r requirements.txt
```

**Dataset Preparation**

1. Create `data` folder inside FastAno_official.
2. Download UCSD Ped2 dataset from this [website](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm) and place it in `data` folder.

**Inference with Pre-trained Weights**

Pre-trained weights are located in `FastAno_official/weights/ped2_best.pth`.

Run inference:

```sh
python main.py
```

### ASTNet

**Setup**

```sh
git clone https://github.com/vt-le/astnet.git
cd ASTNet/ASTNet
pip install -r requirements.txt
```

**Data Preparation**

Prepare dataset structure as described in the `Datasets` section.

**Evaluation**

Download pre-trained models and run:

```sh
python test.py --cfg <path/to/config/file> --model-file <path/to/pre-trained/model>
```

**Training from Scratch**

```sh
python train.py --cfg <path/to/config/file>
```

## Pre-trained Models

Download pre-trained models for evaluation:

- [AIVAD Models](https://drive.google.com/drive/folders/1vSMpDb5jIyc2tNJaYVphguUlFcwPayms?usp=sharing)
- [DMAD Models](https://drive.google.com/drive/folders/1PlRZmTFEQ7_CsrCLP9YI83rvWuAID5DF?usp=sharing)


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
