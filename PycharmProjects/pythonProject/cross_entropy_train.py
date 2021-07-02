import torch
import torchvision
import numpy as np
import typing
import logging
from sklearn.metrics import balanced_accuracy_score
import tqdm


def cross_entropy_train_model( num_epochs: int,
                                  model_steps: int,
                                  train_loader: torch.utils.data,
                                  test_loader : torch.utils.data,
                                  balanced_test_flag: bool,
                                  model: torchvision.models,
                                  optimizer_model: torch.optim,
                                  full_retrain_flag: bool,
                                  epoch_unfreezing: int,
                                  device : str) :

    """
    This function establishes training only the linear classifier of DNN before epoch_unfreezing
    As soon as there is epoch_unfreezing, then the rest part of layers becomes unfixed too simultaneously

    :param num_epochs: the number of epochs for training
    :param model_steps: the number of iterations per one epoch
    :param train_loader: train data loader
    :param test_loader: test data loader
    :param balanced_test_flag : Test sample is balanced -> True, otherwise False
    :param model: Deep neural network (ResNet20, ResNet50, ..)
    :param optimizer_model: Optimizer for DNN (Adam, SGD, .. )
    :param full_retrain_flag: There is training of all layers
    :param epoch_unfreezing: the number of epoch when we unfreeze previous layers for classifier
    :param device: GPU or CPU
    :return: accuracy on train and test correspondingly during the epochs
    """
    accuracy_train = []
    accuracy_validation = []
    model.to(device)

    for epoch in tqdm.tqdm(range(num_epochs)):

        train_acc = 0
        val_acc = 0

        # for validation #
        val_pred_full_logits = []
        val_true_full_logits = []

        if full_retrain_flag and epoch > epoch_unfreezing:
            model.requires_grad_(True)
        else:
            model.requires_grad_(False)
        "-- TO DO--"
        try:
            model.fc.requires_grad_(True)
        except AttributeError as error:
            raise AssertionError
        model.train()

        for num_batch, (batch_x, batch_y) in tqdm.tqdm(zip(range(model_steps), train_loader)):

            logits = model(batch_x.to(device))
            optimizer_model.zero_grad()
            "?? train_loss ??"
            (torch.nn.CrossEntropyLoss()(logits, batch_y.to(device).long())).backward()
            y_pred = logits.max(1)[1].detach().cpu().numpy()
            optimizer_model.step()

            del logits #  ??

            train_acc += np.mean(batch_y.cpu().numpy() == y_pred)

        train_acc /= len(train_loader)
        accuracy_train.append(train_acc)

        with torch.no_grad():
            model.eval()
            for num_batch, (batch_x, batch_y) in tqdm.tqdm(enumerate(test_loader)):

                logits = model(batch_x.to(device))
                "?? validation loss ??"
                y_pred = logits.max(1)[1].detach().cpu().numpy()
                del logits # ??

                if balanced_test_flag:
                    val_acc += np.mean(batch_y.cpu().numpy() == y_pred)
                else:
                    val_pred_full_logits.extend(y_pred)
                    val_true_full_logits.extend([x.item() for x in batch_y.cpu()])

            val_acc = val_acc/len(test_loader) if balanced_test_flag else balanced_accuracy_score(val_true_full_logits,
                                                                                           val_pred_full_logits)
            accuracy_validation.append(val_acc)

    return model, accuracy_train, accuracy_validation
