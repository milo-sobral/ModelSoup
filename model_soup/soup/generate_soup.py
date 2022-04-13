import torch
import os
from enum import Enum
import random


class Methods(Enum):
    UNIFORM = 1
    GREEDY = 2
    PRUNED = 3


def avg_weights(old_weights, new_weights, N):
    for idx, weight_matrix in enumerate(old_weights):
        old_weights[idx] = (N * weight_matrix) + new_weights[idx] / (N + 1)
    return old_weights, N+1


def remove_weights(old_weights, weight_removed, N):
    for idx, weight_matrix in enumerate(old_weights):
        old_weights[idx] = (N * weight_matrix) - weight_removed[idx] / (N - 1)
    return old_weights, N-1


def make_soup(models_folder, method, model_class, evaluator, initial_model_file=None):

    if initial_model_file is None:
        initial_model_file = random.choice(models_folder)

    model_class.load_state_dict(torch.load(initial_model_file))
    model_class.eval()

    final_weights = [param for param in model_class.parameters()]
    N = 1

    if method == Methods.UNIFORM:
        baseline_performance = evaluator.eval_func(model_class)

    files = os.listdir(models_folder)
    files = files.remove(initial_model_file)
    for file in files:
        # TODO: Check that we have a model file and not something else

        # Load model weights into the model class
        model_class.load_state_dict(torch.load(file))
        model_class.eval()

        # Get all weights
        weights = [param for param in model_class.parameters()]

        # Create the soup
        if method == Methods.UNIFORM or method == Methods.PRUNED:
            final_weights, N = avg_weights(final_weights, weights, N)
        elif method == Methods.GREEDY:
            new_performance = evaluator.eval_func(model_class)

            if new_performance >= baseline_performance:
                final_weights, N = avg_weights(final_weights, weights, N)
            baseline_performance = new_performance
     
    if method == Methods.PRUNED:
        baseline_performance = evaluator.eval_func(final_weights)

        for file in os.listdir(models_folder):
            # TODO: Check that we have a model file and not something else

            # Load model weights into the model class
            model_class.load_state_dict(torch.load(file))

            #prune the soup
            final_weights, N = remove_weights(final_weights)
            new_performance = evaluator.eval_func(final_weights)

            #if the performance drops, add back to the soup
            if new_performance < baseline_performance:
                final_weights, N = avg_weights(final_weights, weights, N)
            else:
                baseline_performance = new_performance

    evaluator.set_weights(final_weights)
    return evaluator.get_model()

