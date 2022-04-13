from setuptools import setup

setup(
   name='model_soup',
   version='1.0',
   description='Package to test model soups with Resnets and the CIFAR-100 Dataset',
   packages=['model_soup'],  #same as name
   url='https://github.com/milo-sobral/ModelSoup',
   install_requires=[
       'torch', 
       'torchvision', 
       'argparse'], #external packages as dependencies
)