# Automated_Essay_Scoring

The aim of this project is to develop a model capable of scoring essays taken from a given range of topics.
To reproduce the results we obtained, please follow the following steps:

1. Run [data_preprocessing.ipynb](data_preprocessing.ipynb) to generate the relevant datasets. Don't forget to create a "data" folder in the root to store the processed datasets.

2. You can now train your models by running command lines of type `python train.py --model "ConvNet1D" --depth 0 --seed 1` (best model). Please refer to the `arg_parser` function of the [train.py](train.py) file to get more knowledge on the set of available hyperparameters.

3. [graph.ipynb](graph.ipynb) shows the training and validation performances of each model we trained.

4. [test.ipynb](test.ipynb) shows the test performance of the best model obtained as well as a concrete example of the essay grading task.
