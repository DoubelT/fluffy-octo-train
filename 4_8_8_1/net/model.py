import torch
import torch.nn as nn
from collections import OrderedDict

class BankNodes(nn.Module):
    def __init__(self, config, is_training=True):
        super(BankNodes, self).__init__()
        self.config = config
        self.training = is_training

        # Define the layers using OrderedDict
        layers['fc1'] = nn.Linear(in_features=config["input_size"], out_features=config["hiddenlayer_size1"])
        layers['relu1'] = nn.ReLU()
        layers['fc2'] = nn.Linear(in_features=config["hiddenlayer_size1"], out_features=config["hiddenlayer_size2"])
        layers['relu2'] = nn.ReLU()
        layers['fc3'] = nn.Linear(in_features=config["hiddenlayer_size2"], out_features=config["outputlayer_size"])
        layers['sig3'] = nn.Sigmoid()

        # Create the neural network using the layers
        self.network = nn.Sequential(layers)

    def forward(self, x):
        # Forward pass through the network
        return self.network(x)

if __name__ == "__main__":
    config = {"input_size": 4, "hiddenlayer_size1": 8, "hiddenlayer_size2": 8, "outputlayer_size": 1} # Adjust these values as needed
    m = BankNodes(config)
    x = torch.randn(1, config["input_size"])
    y = m(x)
    print(y)
