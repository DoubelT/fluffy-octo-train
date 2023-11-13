import sys
import xir
import vart
import time
from typing import List
from ctypes import *
import importlib
import numpy as np




def runBankNode(dpu_runner_tfBankNode, input, config):
    config = config

    print("inside the run BankNodes..")
    inputTensors = dpu_runner_tfBankNode.get_input_tensors()  #  get the model input tensor
    outputTensors = dpu_runner_tfBankNode.get_output_tensors() # get the model ouput tensor

    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)

    print(inputTensors[0])
    print(outputTensors[0])

    runSize = 1
    outputData = []

    outputData.append([np.zeros((runSize, outputTensors[0].dims[1]), dtype=np.float32, order="C")])

    shapeIn = (runSize,) + tuple([inputTensors[0].dims[i] for i in range(inputTensors[0].ndim)][1:])
    print('Coded shapeIn: ', shapeIn)

    #print('Input Tensor[0]: ', inputTensors[0])


    inputData = []
    inputData.append(input)


    testOutput = []
    testOutput.append(np.array([4],dtype=np.float32, order="C"))
    testInput = []
    
    #Sample which is a fake so output should be 0
    #testInput.append(np.array([1.6653357,1.0844043,-1.098024,-0.18494527], dtype=np.float32,order="C"))
    
    #Sample which is a orginal so output should be 1
    testInput.append(np.array([-0.6005844,-0.7122524,0.70745796,0.7955938], dtype=np.float32,order="C"))
    
    print("Test Input is: ", testInput)
    print("Test Output is: ", testOutput)
    print("Execute async")
    job_id = dpu_runner_tfBankNode.execute_async(testInput, testOutput) #input output missing !!!
    dpu_runner_tfBankNode.wait(job_id)
    print("Execcution completed..")
    
    sigmoidOutput = sigmoid_rounded(testOutput[0])
    
    print("OUTPUT after running in Output: ",testOutput )
    print("OUTPUT after applying sigmoid_rounded: ",sigmoidOutput )




def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def sigmoid_rounded(x):
    return np.round(sigmoid(x))    



def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."

    root_subgraph = graph.get_root_subgraph() # Retrieves the root subgraph of the input 'graph'
    assert (root_subgraph
            is not None), "Failed to get root subgraph of input Graph object."

    if root_subgraph.is_leaf:
        return [] # If it is a leaf, it means there are no child subgraphs, so the function returns an empty list

    child_subgraphs = root_subgraph.toposort_child_subgraph() # Retrieves a listof child subgraphs of the 'root_subgraph' in topological order
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

    validationset_features = np.loadtxt(r"../dataset/df_validationset_features",delimiter=',')
    validationset_labels = np.loadtxt(r"../dataset/df_validationset_labels", delimiter=',')

    print(validationset_features[0])

    input = validationset_features[0]


    # Measure time
    time_start = time.time()

    """Assigns the runBankNode function with corresponding arguments"""
    print("runBankNode -- main function intialize")
    runBankNode(dpu_runners, input, config)

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
