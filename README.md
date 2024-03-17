# Basic Neural Network on MNIST Dataset

This repository contains a basic implementation of a neural network trained on the MNIST dataset using Python and Numpy (without TensorFlow).

## Overview

The MNIST dataset is a classic benchmark dataset in the field of machine learning and computer vision. It consists of 28x28 grayscale images of handwritten digits (0-9). The task is to classify these images into their respective digit classes.
In this implementation, a simple feedforward neural network is used to classify the images. The architecture of the neural network is as follows:

- Input layer: 784 neurons (28x28 pixels flattened)
- Hidden layer: 10 neurons with ReLU activation
- Output layer: 10 neurons with softmax activation (corresponding to the 10 digit classes)

## Requirements

- Python 3.x
- Numpy 1.26.4
- Pandas 2.2.0

```bash
pip3 install -r requirements.txt
```

## Usage

1. Clone this repository:

```bash
git clone https://github.com/Yash2402/neural-network.git
```

2. Navigate to the project directory:

```bash
cd neural-network
```

Note: After cloning repository, download MNIST Dataset [Mnist Dataset](https://drive.google.com/drive/folders/1pYPgCPVr3MCSMz_lUHxfwzbWViPNnaON?usp=share_link), unzip it and then store the `data/` folder in `neural-network/` folder.

3. Run the `train.py` script to train the neural network:

```bash
python3 train.py [max_iterations] [learning_rate] # max_iterations = 3000 and learning_rate = 0.5 works great for me 
```

4. Test the Neural Network

```bash
python3 test.py
```

Note: To see the next prediction close the matplotlib window

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/Yash2402/neural-network/blob/main/LICENSE) file for details.

## Acknowledgments

This implementation is based on tutorials and resources from Youtube Channels like [3Blue1Brown](https://www.youtube.com/c/3blue1brown), [Samson Zhang](https://www.youtu:be.com/@SamsonZhangTheSalmon) and other online sources.
Feel free to contribute or provide feedback to improve this implementation!
