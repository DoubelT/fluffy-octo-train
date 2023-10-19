import sys
import xir
import vart
import time
from typing import List
from ctypes import *
import random
import pandas as pd
import importlib
import numpy as np



def runBankNode(dpu_runner_tfBankNode, input, config):
    config = config

    print("inside the run BankNodes..")
    inputTensors = dpu_runner_tfBankNode.get_input_tensors()  #  get the model input tensor
    outputTensors = dpu_runner_tfBankNode.get_output_tensors() # get the model ouput tensor
    

    outputHeight = outputTensors[0].dims[1]
    outputWidth = outputTensors[0].dims[2]
    outputChannel = outputTensors[0].dims[3]
    
    outputSize = (outputHeight,outputWidth,outputChannel)
    print("outputSize ", outputSize)
    

    runSize = 1
    shapeIn = (runSize,) + tuple([inputTensors[0].dims[i] for i in range(inputTensors[0].ndim)][1:])
    print("What shapeIn", shapeIn) 
    print("InputTensor[0]: ", inputTensors[0])

    # InputTensor[0]: {name: 'ModelMain__input_0_fix', shape: [1, 416, 416, 3], type: 'xint8', attrs: {'location': 1, 'ddr_addr': 1264, 'bit_width': 8, 'round_mode': 'DPU_ROUND', 'reg_id': 2, 'fix_point': 4, 'if_signed': True}}

    '''prepare batch input/output '''
    outputData = []
    inputData = []
    outputData.append(np.empty((runSize,outputHeight,outputWidth,outputChannel), dtype = np.float32, order = 'C'))
    inputData.append(np.empty((shapeIn), dtype = np.float32, order = 'C'))

    '''init input image to input buffer '''
    inputRun = inputData[0]
    inputRun[0,...] = input.reshape(inputTensors[0].dims[1],inputTensors[0].dims[2],inputTensors[0].dims[3])


    print("Execute async")
    job_id = dpu_runner_tfBankNode.execute_async(inputData,outputData)
    dpu_runner_tfBankNode.wait(job_id)
    print("Execcution completed..")

    print()
    print("Shapes od output: ")
    print(outputData[0].shape) # (1, 13, 13, 255)
    print(outputData[1].shape) # (1, 26, 26, 255)
    print(outputData[2].shape) # (1, 52, 52, 255)
    print()
    print("Input image shape: ", inputData[0].shape) # (1,416,416,3)
    print("Image shape[0]: ", inputData[0][0].shape) # (416,416,3)

    outputData[0] = np.transpose(outputData[0], (0,3,1,2))
    outputData[1] = np.transpose(outputData[1], (0,3,1,2))
    outputData[2] = np.transpose(outputData[2], (0,3,1,2))
  
    
def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."

    root_subgraph = graph.get_root_subgraph() # Retrieves the root subgraph of the input 'graph'
    assert (root_subgraph
            is not None), "Failed to get root subgraph of input Graph object."
    
    if root_subgraph.is_leaf:
        return [] # If it is a leaf, it means there are no child subgraphs, so the function returns an empty list 
    
    child_subgraphs = root_subgraph.toposort_child_subgraph() # Retrieves a list of child subgraphs of the 'root_subgraph' in topological order
    assert child_subgraphs is not None and len(child_subgraphs) > 0

    return [
        # List comprehension that filters the child_subgraphs list to include only those subgraphs that represent DPUs
        cs for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def main(argv):
    
    g = xir.Graph.deserialize(argv[1]) # Deserialize the DPU graph
    subgraphs = get_child_subgraph_dpu(g) # Extract DPU subgraphs from the graph
    assert len(subgraphs) == 1  # only one DPU kernel

    #Creates DPU runner, associated with the DPU subgraph.
    dpu_runners = vart.Runner.create_runner(subgraphs[0], "run")
    print("DPU Runner Created")

    # Get config
    params_path = "param.py"
    config = importlib.import_module(params_path[:-3]).parameters

    # Preprocessing 
    
    df_validationset_features = pd.read_csv(r"../dataset/df_validationset_features")
    df_validationset_labels = pd.read_csv(r"../dataset/df_validationset_labels")
    tensor_validationset_features = torch.tensor(df_validationset_features.values, dtype = torch.float)
    tensor_validationset_labels = torch.tensor(df_validationset_labels.values, dtype = torch.float).view(-1,1)
    
    inputTen = tensor_validationset_features[0]
    
    
    # Measure time 
    time_start = time.time()

    """Assigns the runBankNode function with corresponding arguments"""
    print("runBankNode -- main function intialize")
    runBankNode(dpu_runners, inputTen, config)

    del dpu_runners
    print("DPU runnerr deleted")

    time_end = time.time()
    timetotal = time_end - time_start
    total_frames = 1 
    fps = float(total_frames / timetotal)
    print(
        "FPS=%.2f, total frames = %.2f , time=%.6f seconds"
        % (fps, total_frames, timetotal)
    )

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage : python3 dpu_infer.py <xmodel_file>")
    else:
        main(sys.argv)