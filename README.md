# Automated Essay Scoring

The aim of this project is to develop a model capable of scoring essays taken from a given range of topics.
To reproduce the results we obtained, please follow the following steps:

1. Clone the repository.

2. Go to [https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300](https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300) and download the file entitled GoogleNews-vectors-negative300.bin. Then, put it in a "data" folder located at the root of the directory.

3. Run [data_preprocessing.ipynb](data_preprocessing.ipynb) to generate the relevant datasets.

4. You can now train your models by running command lines of type `python train.py --model "ConvNet1D" --depth 0 --seed 1` (best model). Please refer to the `arg_parser` function of the [train.py](train.py) file to get more knowledge on the set of available hyperparameters.

5. [graphs.ipynb](graphs.ipynb) shows the training and validation performances of each model we trained.

6. [test.ipynb](test.ipynb) shows the test performance of the best model obtained as well as a concrete example of the essay grading task.
