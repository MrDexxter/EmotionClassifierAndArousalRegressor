# Multi-task Affect Recognition Model

![Description of a Deep Learning Model predicting Hu](https://user-images.githubusercontent.com/97404986/235732339-58214308-3577-4b52-b28c-a6aaf5b56712.jpg)

This repository contains code for training and evaluating a multi-task affect recognition model based on deep learning using the PyTorch framework. The model predicts facial expressions, arousal, and valence from facial images and their corresponding landmarks.

## Prerequisites

- Python 3.6 or higher
- PyTorch
- torchvision
- pillow
- numpy
- tqdm
- matplotlib
- scikit-learn
- torchviz

## Installation

To install the required packages, run the following command:

```bash
pip install torch torchvision pillow numpy tqdm matplotlib scikit-learn torchviz
```
## Dataset Preparation
1- Extract the dataset archive and place it in the project directory:
```bash
tar -xf dataset.tar.gz
```
Rename the extracted folder to dataset and ensure that it contains two subfolders: train_set and val_set.
Each of these subfolders should have two additional subfolders: images and annotations.
The images folder should contain the facial images in .jpg format, and the annotations folder should contain the corresponding arousal (.aro.npy), valence (.val.npy), expression (.exp.npy), and landmarks (.lnd.npy) files in .npy format.

## Training
To train the multi-task model, run the following command:

```bash
python train.py
```
This script will train the model and save the best model weights to the models folder.

## Evaluation
To evaluate the trained model on the test dataset, run the following command:

``` bash
python model_testing.ipyb
```
This script will load the saved models, perform predictions on the test dataset, and calculate various evaluation metrics.

## Usage
Use the provided Jupyter Notebook model_testing.ipyb to test the model on your custom images. Update the image paths and landmarks paths in the predict_image function call to test the model with your images.

## Model Visualization
To visualize the model architecture, run the following command:

``` bash
python model_visualization.py
```
This script will generate a diagram of the model architecture and save it as an image file named model_graph_resnet.png.

### License
This project is licensed under the MIT License. See the LICENSE file for more details.
