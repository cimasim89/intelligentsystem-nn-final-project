from pipetools import pipe

from dataset import prepare_dataset, split_dataset
from utils import inspect_wrapper


def inspect_split_dataset(data):
    x_train, x_test, y_train, y_test = data
    print('x_train shape: ', x_train.shape[0])
    print('x_test shape: ', x_test.shape[0])


def train_network(data):
    func = pipe | prepare_dataset() | split_dataset(0.25) | inspect_wrapper(inspect_split_dataset)
    return func(data)
