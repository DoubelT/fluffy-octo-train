# Nerual Networks on the Kria KV260

Contains 8 different neural networks for the execution on the Kria KV260

## Table of Contents

- [Installation](#installation)
- [Structure](#structure)
- [Usage](#usage)

## Installation

Parts of this project can only be executed in within Vitis AI. For installation check [this github repository](https://github.com/Xilinx/Vitis-AI). Then clone this repository inside the Vitis AI folder.

## Structure

- 4_16_1 - 4_8_8_1 contains the neural networks with all the code for exection
- Dataset contains the prepared dataset for the neural networks
- Inference_Result contains the results of the different accuracys of the neural networks during the different stages of the workflow
- JupyterNotebooks contains the code regarding the creation of the neural networks with PyTorch
- diagramms contains an example code on how to generate plots with plotly

## Usage

Reminder: You can only use the AI Quantizer or AI Compiler inside the running docker container of Vitis AI\

1. Using the AI Quantizer: go to inside the directory of the desired network and navigate to `network/quant`. Excute the the `quanter.py` over the command line. Call: `python quanter.py --quant_mode calib` first, then `python quanter.py --quant_mode test --batch_size 1 --deploy`. Inside the same folder will be the `quantize_result` folder which contains the quantized neural network.
2. Using the AI Compiler: root of the folder containing the network. Execute the following command to generate an executable model for the DPU: `vai_c_xir --xmodel ./quant/quantize_result/BankNodes_int.xmodel --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json --net_name BankNodes --output_dir ./Compiled_Banknodes`
3. Execute the generated model on the DPU: clone the whole repository on the KV260 first. navigate to `quant_infer` directory inside the network directory. Execute the python file `quant_inferenc.py` like this: `python quant_inferenc.py ../Compiled_Banknodes/BankNodes.xmodel`


