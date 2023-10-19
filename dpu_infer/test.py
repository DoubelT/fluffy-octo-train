import sys
import xir
import vart
import time
from typing import List
from ctypes import *
import random
import pandas as pd
import importlib


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


    inputTensor = dpu_runners.get_input_tensors()
    outputTensor = dpu_runners.get_output_tensors()
    
    print("Input Tensor print: ", inputTensor)
    print("Output Tensor print: ", outputTensor)
    
    input_dim = tuple(inputTensor[1].get_tensor().dims)
    batch = input_ndim[0]
    width = input_ndim[1]
    height = input_ndim[2]
    
    print("Input Tensor Tuple: ", input_dim)
    print("Batch: ", batch)
    print("Width: ", width)
    print("Height: ", height)
    
    
    
    
    
    # Measure time 
    time_start = time.time()



    del dpu_runners
    print("DPU runnerr deleted")


if __name__ == "__main__":
    #if len(sys.argv) != 2:
    #    print("usage : python3 test.py <xmodel_file>")
   # else:
        main(sys.argv)