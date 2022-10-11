import json
import os

from util.list_utils import dict_mean


def average_training_metrics(metrics: list, save_dir: str = None, print_metrics: bool = True):
    """
    Averages the training metrics of a list of models.
    :param metrics: A list of training metrics.
    :param save_dir: The directory to save the metrics to.
    :param print_metrics: Whether to print the metrics.
    :return: The average training metrics.
    """

    lowest_loss = float('inf')
    highest_f1 = float('-inf')
    for d in metrics:
        if d['lowest_loss'] < lowest_loss:
            lowest_loss = d['lowest_loss']
        if d['highest_f1'] > highest_f1:
            highest_f1 = d['highest_f1']

        d.pop('best_epoch')
        d.pop('lowest_loss')
        d.pop('lowest_loss_epoch')
        d.pop('highest_f1')
        d.pop('highest_f1_epoch')

    # Print the metrics
    print("Training metrics:")
    mean = dict_mean(metrics)
    mean['lowest_loss'] = lowest_loss
    mean['highest_f1'] = highest_f1

    if print_metrics:
        print(mean)

    if save_dir is not None:
        with open(os.path.join(save_dir, 'train_metrics.txt'), 'w') as f:
            json.dump(mean, f, indent=4)

    return mean


def average_test_metrics(metrics: list, save_dir: str = None, print_metrics: bool = True):
    """
    Averages the test metrics of a list of models.
    :param metrics: A list of test metrics.
    :param save_dir: The directory to save the metrics to.
    :param print_metrics: Whether to print the metrics.
    :return: The average test metrics.
    """

    # Print the metrics
    print("Test metrics:")
    mean = dict_mean(metrics)

    f1 = [d['f1'] for d in metrics]
    mean['best_f1'] = f1
    mean['best_f1_model'] = f1.index(max(f1))

    if print_metrics:
        print(mean)
        print('Best Test F1 model: {}'.format(mean['best_f1_model']))

    if save_dir is not None:
        with open(os.path.join(save_dir, 'test_metrics.txt'), 'w') as f:
            json.dump(mean, f, indent=4)

    return mean
