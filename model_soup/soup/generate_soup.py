import torch
import os
from enum import Enum
import random
from copy import deepcopy


class Methods(Enum):
    UNIFORM = 1
    GREEDY = 2
    PRUNED = 3


def add_ingradient(soup, path, N):
    ingradient = deepcopy(soup)
    ingradient.load_state_dict(torch.load(path)['state_dict'])
    ingradient.cuda()

    for param1, param2 in zip(soup.parameters(), ingradient.parameters()):
        param1.data = ((param1.data * N) + param2.data) / (N+1)
        
    return soup, N+1


def make_soup(models_folder, soup, evaluator, device=None, method=Methods.GREEDY, initial_model_file=None):
    '''
    Generates a soup using the given evaluator and method, returns soup and best performance
    Inputs:
        - models_folder : [required] folder containing pytorch trained models to use
        - soup : [required] Pytorch Class of the desired model
        - evaluator : [required] An evaluator object which will be used to evaluate the performance of the model
        - device : Torch device to run inference
        - method : One of Greedy or Uniform
        - initial_model_file : specify which file to use initially
    Returns: 
        - soup : The soup made according to the method specified
        - final_performance : The final performance according to the specified evaluator of the model
        - N : Number of ingradients used in the final soup
    '''

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    all_model_files = os.listdir(models_folder)

    while initial_model_file is None:
        chosen_file = random.choice(all_model_files)
        if os.path.isfile(os.path.join(models_folder, chosen_file)): #ignore hidden directories
            initial_model_file = chosen_file

    soup.load_state_dict(torch.load(os.path.join(models_folder, initial_model_file), map_location=device)['state_dict'])
    soup.to(device)
    soup.eval()

    N = 1

    if method == Methods.GREEDY:
        baseline_performance = evaluator.eval_func(soup)
    print(f"baseline: {baseline_performance}")

    all_model_files.remove(initial_model_file)
    for file in all_model_files:
        if os.path.isfile(os.path.join(models_folder, file)): #ignore hidden directories
            file = os.path.join(models_folder, file)
            soup_next = deepcopy(soup)
            soup_next, N = add_ingradient(soup_next, file, N)
            new_performance = evaluator.eval_func(soup_next)
            print(f"new perf: {new_performance}")

            if method == Methods.GREEDY:
                if new_performance >= baseline_performance:
                    soup = soup_next
                    baseline_performance = new_performance
                else:
                    N -= 1
            elif method == Methods.UNIFORM:
                soup = soup_next
            else:
                raise NotImplemented                     

    final_performance = evaluator.eval_func(soup)
    return soup, final_performance, N

