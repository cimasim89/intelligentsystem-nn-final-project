from pipetools import pipe
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from dataset import prepare_dataset, split_dataset
from utils import inspect_wrapper


def inspect_split_dataset(data):
    x_train, x_test, y_train, y_test = data
    print(f'x_train shape: {x_train.shape[0]}')
    print(f'x_test shape: {x_test.shape[0]}')
    print(f'Features extracted: {x_train.shape[1]}')


def get_model(alpha, batch_size, epsilon, hidden_layer_sizes,
              learning_rate, max_iter):
    return MLPClassifier(alpha=alpha, batch_size=batch_size, epsilon=epsilon, hidden_layer_sizes=hidden_layer_sizes,
                         learning_rate=learning_rate, max_iter=max_iter)


def train(alpha, batch_size, epsilon, hidden_layer_sizes,
          learning_rate, max_iter):
    def inner(data):
        model = get_model(alpha=alpha, batch_size=batch_size, epsilon=epsilon, hidden_layer_sizes=hidden_layer_sizes,
                          learning_rate=learning_rate, max_iter=max_iter)
        x_train, x_test, y_train, y_test = data
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        print("Accuracy: {:.2f}%".format(accuracy * 100))

    return inner


def train_network(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,),
                  learning_rate='adaptive', max_iter=500, test_size=0.2, random_state=9):
    return pipe \
           | prepare_dataset() \
           | split_dataset(test_size=test_size, random_state=random_state) \
           | inspect_wrapper(inspect_split_dataset) \
           | train(alpha=alpha, batch_size=batch_size, epsilon=epsilon, hidden_layer_sizes=hidden_layer_sizes,
                   learning_rate=learning_rate, max_iter=max_iter)
