import numpy as np
from sklearn.model_selection import train_test_split

from utils import curr_reduce


def filter_dataset(emotions):
    def inner(data):
        return data.get("emotion") in emotions

    return inner


def prepare_dataset_reducer(accumulator, data):
    x, y = accumulator
    return x + [data.get("features")], y + [data.get("emotion")]


def prepare_dataset():
    return curr_reduce(prepare_dataset_reducer)(([], []))


def split_dataset(test_size=0.2, random_state=9):
    def inner(data):
        x, y = data
        return train_test_split(np.array(x), y, test_size=test_size, random_state=random_state)

    return inner
