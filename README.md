# Deciphering Deep Sea Scrolls

This project focusses on the segmentation and recognition of ancient Hebrew scrolls. We implemented a pipeline which made use of connected component analysis (CCA) for segmenting the letters, followed by ResNet50 and AlexNet for classifying the characters. The goal was to build a pipeline for processing and classifying hard to read historical handwritten text.

## Installation & Running the pipeline

Step 1. Clone the repository

```bash
git clone https://github.com/TexMcGinley/DeepLearningPractical
cd DeepLearningPractical
```

Step 2. Create a virtual environment. Code was created with python 3.11.

```bash
conda create --name dlp python=3.11
conda activate dlp
```

Step 3. Install dependencies

Note: This only installs the CPU version of PyTorch, the CUDA version is recommended.

```bash
pip install .
```

Step 4. Run the base model (ResNet50)

```bash
python main.py /path/to/input
```

### Optional

Run model with AlexNet. The AlexNet weights need to be downloaded from [here](https://drive.google.com/drive/folders/1IG2JCnTzJKKXvYyTGU_rZ6KwEtjc6-L0?usp=sharing) and put in the models folder as follows.

```
├── README.md
├── build
├── data
├── main.py
├── models
│   ├── alexnet_model.pth       <---- Put the AlexNet weights here
│   └── resnet50_model.pth
├── outputs
├── plots
├── requirements.txt
├── results
├── scripts
├── setup.py
└── src
```

To run with AlexNet

```bash
python main.py /path/to/input --model alexnet
```

The intermediair results are in the `outputs` folder while the final model predictions per input are in the `results` folder. To run the evaluation script make sure the ground truth files follow the following labeling scheme `img_001.txt, img_002.txt...`.

To run with evaluation

```bash
python main.py /path/to/input --model alexnet --answers /path/to/answers
```

To images used to train the model can be downloaded [here](https://drive.google.com/file/d/1Ky6vJA1Dw_zW1TT_UAycnWb43EZCJsd5/view?usp=sharing). The images need to be and put in the data folder as follows:

```
├── README.md
├── build
├── data
│   ├── test
│   └── train                   <---- Unzipped train file
├── main.py
├── models
├── outputs
├── plots
├── requirements.txt
├── results
├── scripts
├── setup.py
└── src
```
