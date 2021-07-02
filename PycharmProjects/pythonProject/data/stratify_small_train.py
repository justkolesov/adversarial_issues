import torch
import tqdm
import random
import numpy as np

def stratify_small_train_data(train_data,
                              size_of_train: int) -> list:

    """
    This function makes stratified sample of training data
    with desired numbers of objects per class

    :param train_data: full training data balanced or unbalanced
    :param size_of_train: the number of objects per class
    :return: stratified sample from training data
    """

    "?? typing for data ??"

    new_data = []
    mask = [[] for _ in range(len(train_data.classes))]
    for idx in range(len(train_data)):
        mask[train_data[idx][1]].append(idx)

    "?? more beautiful for shuffling ??"
    mask = [np.random.permutation(np.array(certain_mask)) for certain_mask in mask]

    for cl in range(len(train_data.classes)):
        new_data.extend( [(train_data[mask[cl][idx]][0], cl) for idx in range(size_of_train)] )

    return new_data



