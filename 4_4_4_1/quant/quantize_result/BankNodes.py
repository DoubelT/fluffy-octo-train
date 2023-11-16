# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class BankNodes(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(BankNodes, self).__init__()
        self.module_0 = py_nndct.nn.Input() #BankNodes::input_0
        self.module_1 = py_nndct.nn.Linear(in_features=4, out_features=4, bias=True) #BankNodes::BankNodes/Sequential[network]/Linear[fc1]/input.1
        self.module_2 = py_nndct.nn.ReLU(inplace=False) #BankNodes::BankNodes/Sequential[network]/ReLU[relu1]/48
        self.module_3 = py_nndct.nn.Linear(in_features=4, out_features=4, bias=True) #BankNodes::BankNodes/Sequential[network]/Linear[fc2]/input
        self.module_4 = py_nndct.nn.ReLU(inplace=False) #BankNodes::BankNodes/Sequential[network]/ReLU[relu2]/50
        self.module_5 = py_nndct.nn.Linear(in_features=4, out_features=1, bias=True) #BankNodes::BankNodes/Sequential[network]/Linear[fc3]/51
        self.module_6 = py_nndct.nn.Sigmoid() #BankNodes::BankNodes/Sequential[network]/Sigmoid[sig3]/52

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_0 = self.module_5(output_module_0)
        output_module_0 = self.module_6(output_module_0)
        return output_module_0
