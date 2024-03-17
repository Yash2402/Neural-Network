# Basic Neural Network on MNIST Dataset
A deep neural network model with one hidden layer that can classify digits from 0-9 (MNIST Dataset). This Neural Network contains 3 layer [284, 10, 10].

# Run
```bash
git clone https://github.com/Yash2402/neural-network.git
```
After cloning repository, download MNIST Dataset [Mnist Dataset](https://drive.google.com/drive/folders/1pYPgCPVr3MCSMz_lUHxfwzbWViPNnaON?usp=share_link), unzip it and then store the ```data/``` folder in ```neural-network/``` folder.

## Train the Neural Network
```bash
cd neural-network
pip3 install -r requirements.txt
python3 train.py [max_iterations] [learning_rate] # max_iterations = 3000 and learning_rate = 0.5 works great for me 
```
## Test the Neural Network
```bash
python3 test.py
```
