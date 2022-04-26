import torch
import os
from enum import Enum
import random
from copy import deepcopy


class Methods(Enum):
    UNIFORM = 1
    GREEDY = 2
    PRUNED = 3


class Strategy(Enum):
    RANDOM = 1
    SORTED = 2


def add_ingradient(soup, path, N):
    ingradient = deepcopy(soup)
    ingradient.load_state_dict(torch.load(path)['state_dict'])
    ingradient.cuda()

    for param1, param2 in zip(soup.parameters(), ingradient.parameters()):
        param1.data = ((param1.data * N) + param2.data) / (N+1)
        
    return soup, N+1

def remove_ingradient(soup, path, N):
    ingradient = deepcopy(soup)
    ingradient.load_state_dict(torch.load(path)['state_dict'])
    ingradient.cuda()

    for param1, param2 in zip(soup.parameters(), ingradient.parameters()):
        param1.data = ((param1.data * N) - param2.data) / (N-1)
        
    return soup, N-1


def make_soup(models_folder, soup, evaluator, num_ingradients=0, num_passes=1, device=None, method=Methods.GREEDY, initial_model_file=None, strategy=Strategy.RANDOM):
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

    # if strategy == Strategy.SORTED:
    models = {}
    for model in all_model_files:
        if os.path.isfile(os.path.join(models_folder, model)):
            soup.load_state_dict(torch.load(os.path.join(models_folder, model), map_location=device)['state_dict'])
            soup.to(device)
            soup.eval()
            performance = evaluator.eval_func(soup)
            if performance >= 2: #do not add the models that were unable to learn
                models[model] = performance
    models = dict(sorted(models.items(), key=lambda item: item[1], reverse=True))
    all_model_files = list(models.keys())
    initial_model_file = all_model_files[0]


    if strategy == Strategy.RANDOM:
        random.shuffle(all_model_files)

    # while initial_model_file is None:
    #     chosen_file = random.choice(all_model_files)
    #     if os.path.isfile(os.path.join(models_folder, chosen_file)): #ignore hidden directories
    #         initial_model_file = chosen_file

    soup.load_state_dict(torch.load(os.path.join(models_folder, initial_model_file), map_location=device)['state_dict'])
    soup.to(device)
    soup.eval()

    N = 1

    if method == Methods.GREEDY:
        baseline_performance = evaluator.eval_func(soup,'valid')
    # print(f"baseline: {baseline_performance}")

    all_model_files.remove(initial_model_file)
    for iteration in range(num_passes):
        models_list = deepcopy(all_model_files)
        for file in models_list:
            if os.path.isfile(os.path.join(models_folder, file)): #ignore hidden directories
                file = os.path.join(models_folder, file)
                soup_next = deepcopy(soup)
                soup_next, N = add_ingradient(soup_next, file, N)
                new_performance = evaluator.eval_func(soup_next,'valid')
            

                if method == Methods.GREEDY:
                    print(f"new perf: {new_performance}")
                    if new_performance >= baseline_performance:
                        soup = soup_next
                        print("added model into soup!")
                        all_model_files.remove(file) #remove this model from the considered models for the next passes
                        baseline_performance = new_performance
                        if num_ingradients != 0:
                            if N >= num_ingradients:
                                break
                    else:
                        N -= 1
                elif method == Methods.UNIFORM or method == Methods.PRUNED:
                    soup = soup_next
        if method == Methods.UNIFORM or method == Methods.PRUNED: #only continue the passes if greedy soup
            break

    
    if method == Methods.PRUNED:
        baseline_performance = evaluator.eval_func(soup,'valid')
        print(f"baseline (uniform soup): {baseline_performance}")
        for iteration in range(num_passes):
            models_list = deepcopy(all_model_files)
            for file in reversed(models_list):
                if os.path.isfile(os.path.join(models_folder, file)): #ignore hidden directories
                    file = os.path.join(models_folder, file)
                    soup_next = deepcopy(soup)
                    soup_next, N = remove_ingradient(soup_next, file, N)
                    new_performance = evaluator.eval_func(soup_next,'valid')
                    print(f"new perf: {new_performance}")

                    if new_performance >= baseline_performance:
                        soup = soup_next
                        print("removed model from soup!")
                        print(file)
                        print(all_model_files)
                        print(models_list)
                        all_model_files.remove(file) #remove this model from the considered models for the next passes
                        baseline_performance = new_performance
                    else:
                        N += 1

               

    final_performance = evaluator.eval_func(soup,'test')
    return soup, final_performance, N

