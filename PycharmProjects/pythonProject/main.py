import argparse
import logging
import sys

import torch
import torchvision
import numpy as np
import typing

import sklearn
import tqdm
import collections

from sklearn.metrics import balanced_accuracy_score

from data.load import load_datasets
from data.stratify_small_train import stratify_small_train_data
from data.dataloader import task_loaders_function
from cross_entropy_train import cross_entropy_train_model
from SKD.SKD_from_ImageNet import *
from models.ResNet_models import *
from utils.ResNet_filters_initialization import *
from utils.train_loader_SKD import create_SKD
from utils.discriminators import create_discriminators_resnet50
from adversarial_train import train_CNN
from utils.plot_learning_curves import plot_plot
import configs


if __name__ == "__main__":

    logger = logging.getLogger("Adversarial logger")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(stdout_handler)
    logger.setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument("--name_of_dataset", type=str, default="stanford_dogs")
    parser.add_argument("--size_of_train", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--device_discr", type=str, default="cpu")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device",type=str,default="cpu")

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--model_steps", type=int, default = 25)
    parser.add_argument("--max_gen_epochs",type=int, default= 1)
    parser.add_argument("--max_gen_draws", type=int, default=1)
    parser.add_argument("--discriminator_steps", type=int,default=100)
    parser.add_argument("--epoch_unfreezing",type=int,default=2)
    parser.add_argument("--balanced_test_flag",type=bool,default=False)
    parser.add_argument("--full_retrain_flag",type=bool,default=False)


    args = parser.parse_args()

    # data #
    train_data, test_data, classes = load_datasets(args.name_of_dataset)
    train_small_data = stratify_small_train_data(train_data, args.size_of_train)
    task_loaders = task_loaders_function(train_small_data, test_data, args.batch_size)

    # SKD #
    "how to accurately download it : wget or how else"
    weights = torch.load("./moco_v2_800ep_pretrain.pth.tar")
    SKD_convolutional_weights = get_filters(weights)


    # model #
    model = ResNet50(args.pretrained)
    model = ResNet50_filters_from_Imagenet_init(model, len(train_data.classes) )


    # train loaders SKD #
    train_loaders_SKD, ker_iters_gen = create_SKD(SKD_convolutional_weights, model)

    # discriminators
    discriminators = create_discriminators_resnet50(model, args.device_discr)
    LR_dict = {name: lr for name, lr in zip(SKD_convolutional_weights.keys(), configs.learning_rate_discriminators)}
    optimizer_discriminator = {name: torch.optim.Adam(model.parameters(), lr=LR_dict[name])
                                 for name, model in discriminators.items()}

    # optimizer_generator
    optimizer_generator = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    #beta dictionary
    beta_dictionary = {name: beta for name, beta in zip(SKD_convolutional_weights.keys(), configs.BETA_LIST)}
    batch_size_SKD = {name: batch for name, batch in zip(SKD_convolutional_weights.keys(), configs.batch_skd)}
    batch_sizes = batch_size_SKD
    logger.warning("start to train by adversarial approach")

    #train model adversarially

    model,history = train_CNN(args.num_epochs,
                             args.epoch_unfreezing,
              args.max_gen_epochs,
              args.model_steps,
              args.max_gen_draws,
              model,
              discriminators,
              task_loaders,
              beta_dictionary,
              optimizer_generator,
              optimizer_discriminator,
              args.discriminator_steps,
              ker_iters_gen,
              train_loaders_SKD,
              logger,
              args.device)


    #model for cross-entropy
    model_ce = ResNet50(args.pretrained)
    model_ce = ResNet50_filters_from_Imagenet_init(model_ce, len(train_data.classes))

    #optimizer for model cross-entropy
    optimizer_generator_ce = torch.optim.Adam(
        model_ce.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    logger.warning("start train by cross-entropy")
    model_ce, accuracy_train_ce, accuracy_validation_ce = cross_entropy_train_model(args.num_epochs,
                                                                            args.model_steps,
                                                                            task_loaders['train'],
                                                                            task_loaders['test'],
                                                                            args.balanced_test_flag,
                                                                            model_ce,
                                                                            optimizer_generator_ce,
                                                                            args.full_retrain_flag,
                                                                            args.epoch_unfreezing,
                                                                            args.device)
    plot_plot(history['acc']['val'], accuracy_train_ce)

























