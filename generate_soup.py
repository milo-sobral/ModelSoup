# Take a folder full of different models as input
# generate a model that is the average using one of the specified methods
# Outputs this model

import argparse
import torch
import os

METHODS = ['UNIFORM', 'GREEDY', 'PRUNED']

def avg_weights(old_weights, new_weights, N):
    for idx, weight_matrix in enumerate(old_weights):
        old_weights[idx] = (N * weight_matrix) + new_weights[idx] / (N + 1)
    return old_weights, N+1

def remove_weights(old_weights, weight_removed, N):
    for idx, weight_matrix in enumerate(old_weights):
        old_weights[idx] = (N * weight_matrix) - new_weights[idx] / (N - 1)
    return old_weights, N-1


def make_soup(models_folder, method, model_class, initial_model_file, eval_func=None):

    model_class.load_state_dict(torch.load(initial_model_file))
    model_class.eval()

    final_weights = [param for param in model_class.parameters()]
    N = 1

    if method == "GREEDY":
        baseline_performance = eval_func(model_class)

    for file in os.listdir(models_folder):
        # TODO: Check that we have a model file and not something else

        # Load model weights into the model class
        model_class.load_state_dict(torch.load(file))
        model_class.eval()

        # Get all weights
        weights = [param for param in model_class.parameters()]

        # Create the soup
        if method == "UNIFORM" or method == "PRUNED":
            final_weights, N = avg_weights(final_weights, weights, N)
        elif method == "GREEDY":
            new_performance = eval_func(model_class)

            if new_performance >= baseline_performance:
                final_weights, N = avg_weights(final_weights, weights, N)
            baseline_performance = new_performance
     
    if method == "PRUNED":
        baseline_performance = eval_func(final_weights)

        for file in os.listdir(models_folder):
            # TODO: Check that we have a model file and not something else

            # Load model weights into the model class
            model_class.load_state_dict(torch.load(file))

            #prune the soup
            final_weights, N = remove_weights(final_weights)
            new_performance = eval_func(final_weights)

            #if the performance drops, add back to the soup
            if new_performance < baseline_performance:
                final_weights, N = avg_weights(final_weights, weights, N)
            else:
                baseline_performance = new_performance

                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('models_folder')
    parser.add_argument('initial_model')
    parser.add_argument('--method', type=str, choices=METHODS)  
    args = parser.parse_args()
    make_soup(args.models_folder, args.method, args.initial_model)
