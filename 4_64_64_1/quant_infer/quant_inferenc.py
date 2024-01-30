import torch
import torch.nn as nn
import numpy as np
import importlib
import pandas as pd

import pytorch_nndct

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = torch.jit.load('../quant/quantize_result/BankNodes_int.pt', map_location=torch.device('cpu'))

model.eval()
model = model.to(device)
print(model)


####### Load the config 
params_path = "param.py"
config = importlib.import_module(params_path[:-3]).parameters


####### Load sets

df_validationset_features = pd.read_csv(r"../dataset/df_validationset_features")
df_validationset_labels = pd.read_csv(r"../dataset/df_validationset_labels")
tensor_validationset_features = torch.tensor(df_validationset_features.values, dtype = torch.float)
tensor_validationset_labels = torch.tensor(df_validationset_labels.values, dtype = torch.float).view(-1,1)
#print(tensor_validationset_features)
#print(tensor_validationset_labels)


####### Inference

# Perform inference
with torch.no_grad():
    outputs = torch.round(model(tensor_validationset_features))
    print("Output:", outputs)
    print("Expect:", tensor_validationset_labels)
    eq_tensor = torch.eq(outputs, tensor_validationset_labels)
    print(eq_tensor)
    count_true = (eq_tensor == True).sum().item()
    print("True predictions: ",count_true)
    print("Accuracy: ", count_true/eq_tensor.size()[0])

print("#####DONE#####")

   
