import torch
import torch.nn as nn
import importlib
import random
import pandas as pd

import sys
import os

# Projekt Ordner zum Python path hinzufügen
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, '..'))

# Importieren des models
from net.model import BankNodes


# GPU auswählen wenn diese zur Verfügung steht
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameter laden
param_path = "param.py"
config = importlib.import_module(param_path[:-3]).parameters


# Model erstellen
model = BankNodes(config, is_training=False)
model.train(False)

# Model auf mehreren GPUs laufen lassen

#model = nn.DataParallel(model) aktivieren wenn das Model auch für mehrer GPUs etc gemacht worden ist
model = model.to(device)


# Vortrainiertes Netwerk laden

if config["pretrain_snapshot"]:
    state_dict = torch.load(config["pretrain_snapshot"], map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
else:
    raise Exception("missing pretrain_snapshot!!!")

print(model)

# Load validationset
df_validationset_features = pd.read_csv("../dataset/df_validationset_features")
df_validationset_labels = pd.read_csv("../dataset/df_validationset_labels")
tensor_validationset_features = torch.tensor(df_validationset_features.values, dtype = torch.float)
tensor_validationset_labels = torch.tensor(df_validationset_labels.values, dtype = torch.float).view(-1,1)

# Starte Inferenc
with torch.no_grad():
    model.eval()
    validation_outputs = model(tensor_validationset_features)
    validation_outputs = (validation_outputs>0.5).float()
    network_accuracy = (validation_outputs == tensor_validationset_labels).float().mean()
    print("Network Accuracy: ", network_accuracy.item()*100,"%")