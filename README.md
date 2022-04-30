# ModelSoup

* Authors:
  * Milo Sobral
  * Charles Dansereau
  * Maninder Bhogal
  * Mehdi Zalai
* Contact us: 
  * {milo.sobral, charles.dansereau, maninder.bhogal, mehdi.zalai}@polymtl.ca

This repository contains the code for our study on the benefits and limitations of model soups. 

### Abstract
In this paper, we compare Model Soups performances on three different models (ResNet, ViT and EfficientNet) using three Soup Recipes (Greedy Soup Sorted, Greedy Soup Random and Uniform soup), and reproduce the results of the authors. We then introduce a new Soup Recipe called Pruned Soup. Results from the soups were better than the best individual model for the pre-trained vision transformer, but were much worst for the ResNet and the EfficientNet. Our pruned soup performed better than the uniform and greedy soups presented in the original paper. We also discuss the limitations of weight-averaging that were found during the experiments.

## Notebooks
* Notebook used for [training of the ResNet from scratch](https://colab.research.google.com/drive/1D_ucvp5OiaWGEho3ATwL9M6I0LCnpEZY)
* Notebook used to test [model soups](https://colab.research.google.com/drive/1yyRSK9x35gErpMy_LQjB4ULR8GagSVVJ?usp=sharing)
* Notebook used to [fine-tune vision models](https://colab.research.google.com/drive/13nYqc5F9L5WVBy3mRYlZBGCf0ekENYMk?usp=sharing)

## Getting Started

Clone and install the library using the following: 
* `git clone [this repo]`
* `cd ModelSoup && pip install -e .`

You must then import the desired functions using:
* `from model_soup.model_evaluators.cifar_eval import CIFAR100Evaluator`
* `from model_soup.model_evaluators.resnet import resnet20`
* `from model_soup.soup.generate_soup import make_soup, Methods, Strategy`

The make_soup function needs both an instance of the model and an evaluator so initialize them like so:
* `model = resnet20()`
* `evaluator = CIFAR100Evaluator()`

You can then call the function with the following arguments:
* `final_model, performance, N = make_soup(models_folder=MODEL_PATH, `
  * `soup=model,`
  * `evaluator=evaluator,`
  * `method=Methods.GREEDY,`
  * `strategy=Strategy.RANDOM,`
  * `num_ingradients=0)`

The outputs are:
* final_model: a pytorch model with the best performing soup.
* performance: The final performance of the soup according to the given evaluator.
* N: the number of ingredients used in the final soup.
### Implementation details
The model soup code has two main components:
* The generate_soup.py script which contains the main make_soup() function which uses pytorch trained models and an evaluator to generate different kinds of soups.
* An evaluator interface, which must be implemented by the Evaluator object used to generate the soups. An example of an Evaluator Class can be found in model_evaluators/cifar_eval. 
The evaluator provides the metric upon which the classification of the models and the selection of ingredients for the soup will be made. The code is designed to always maximize that metric so we must be careful when using metrics like loss that are designed to be minimized. The Evaluator is responsible for loading the desired dataset and iterating over it. The soup code is designed to use two separate sets: one for initial ordering of the models and for selecting ingredients and one for computing final performance. This distinction is critical to avoid overfitting our soups to one single set. Make sure to provide this in the given Evaluator. 
