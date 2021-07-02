from collections import OrderedDict
from collections import defaultdict
import tqdm
from tqdm import notebook
import time
import torch
import torchvision
import numpy as np

import utils.plot_learning_curves
from utils.discriminators import dis_loss, gen_loss_dis
from utils.plot_learning_curves import protect
from sklearn.metrics import balanced_accuracy_score


def train_CNN(num_epochs,
              epoch_unfreezing,
              max_gen_epochs,
              generator_steps,
              max_gen_draws,
              model,
              discriminators,
              task_loaders,
              beta_dictionary,
              optimizer_generator,
              optimizer_discriminator,
              discriminator_steps,
              ker_iters_gen,
              train_loaders_SKD,
              logger,
              device):
    """
    num_epochs : num_epochs of CNN
    max_gen_epochs : amount epochs of generator (by default 1)
    generator_steps: amount steps of pseudo-generator
    max_gen_draws :
    model : CNN
    discriminators : dictionary of discriminators
    task_loaders : dictionary of train loaders of CIFAR10 for CNN
    beta_dictionary : dictionary of constants of pseudo-generator loss
    optimizer_generator: optimizer of CNN
    optimizer_discriminator : optimizer of a discriminator
    discriminator_steps : amount steps of discriminator
    ker_iters_gen: generator for a batch of current convolutional filters in a layer

    """

    history = defaultdict(lambda: defaultdict(list))

    betas_history = {layer: [value] for layer, value in beta_dictionary.items()}

    accuracy_train = []
    accuracy_val = []

    for epoch in tqdm.tqdm(range(num_epochs)):

        train_loss = 0
        train_acc = 0
        loss_task_sum = 0
        gen_loss_task_sum = 0

        val_loss = 0
        val_acc = 0

        start_time = time.time()

        # training of model

        for name, dis in discriminators.items():
            dis.cpu()

        model.train()
        if epoch > epoch_unfreezing:
            model.requires_grad_(True)

        else:
            model.requires_grad_(False)
            model.fc.requires_grad_(True)

        # all discriminators eval
        for discriminator in discriminators.values():
            discriminator.requires_grad_(False)

        logger.warning("start_train")

        for generator_epoch in tqdm.tqdm(range(max_gen_epochs)):
            for num_batch, (batch_x, batch_y) in zip(range(generator_steps), task_loaders['train']):

                model.to(device)
                logits = model(batch_x.to(device))  #
                optimizer_generator.zero_grad()
                if epoch > epoch_unfreezing:
                    torch.nn.CrossEntropyLoss()(logits, batch_y.to(device).long()).backward(
                        retain_graph=True)  # task loss on cuda
                else:
                    torch.nn.CrossEntropyLoss()(logits, batch_y.to(device).long()).backward()

                y_pred_train = logits.max(1)[1].detach().cpu().numpy()
                loss_task = torch.nn.CrossEntropyLoss()(logits.cpu(), batch_y.cpu().long())
                del logits
                train_acc += np.mean(batch_y.cpu().numpy() == y_pred_train)

                # loss_task = gen_loss_task(model, input=batch_x, target = batch_y.long())

                # calculate pseudo-generator loss throughout all layers

                if epoch > epoch_unfreezing:
                    dis_terms = {
                        (layer): gen_loss_dis(dis, fake=next(ker_iters_gen[layer]).view(-1, \
                                                                                        next(
                                                                                            ker_iters_gen[layer]).shape[
                                                                                            -1] * \
                                                                                        next(
                                                                                            ker_iters_gen[layer]).shape[
                                                                                            -2]).cpu())
                        for layer, dis in discriminators.items()

                    }

                    # calculate sum loss : CrossEntropyLoss + pseudo-generator loss of each layer
                    value = sum(
                        beta_dictionary.get(layer) * term
                        for layer, term in dis_terms.items())

                    value.to(device).backward()

                optimizer_generator.step()
                # model.cpu()

                # to get accuracy let's calculate predictions of the CNN

                train_loss += np.sum((loss_task).detach().cpu().numpy())
                # loss_task_sum += np.sum(loss_task.detach().cpu().numpy())
                # gen_loss_task_sum += np.sum(value.detach().cpu().numpy())

        # normalizing train_loss and train_acc
        train_loss /= len(task_loaders['train'])
        train_acc /= len(task_loaders['train'])

        accuracy_train.append(train_acc)

        history['loss']['train'].append(train_loss)
        history['acc']['train'].append(train_acc)
        history['task_loss']['train'].append(loss_task_sum)
        history['gen_loss_task']['train'].append(gen_loss_task_sum)
        logger.warning("finish train")

        # training of discriminator
        # model's gradient doesn't backprop

        model.cpu()
        if epoch > epoch_unfreezing:
            for name, dis in discriminators.items():
                dis.to(device)
            for discriminator in discriminators.values():
                discriminator.requires_grad_(True)

            for _ in tqdm.tqdm(range(discriminator_steps)):
                loss = {}
                for layer, dis in discriminators.items():

                    # get a batch of SKD filters and a batch of current convo filters
                    real = next(iter(train_loaders_SKD[layer]))
                    fake = next(ker_iters_gen[layer]).detach()

                    # to align batch sizes when BATCH_SIZE_SKD != BATCH_SIZE_CONVO
                    if real.shape[0] < fake.shape[0]:
                        fake = fake[:real.shape[0]]
                    elif real.shape[0] > fake.shape[0]:
                        real = real[:fake.shape[0]]

                    # calculate loss of dsicriminator in a batch
                    loss[layer] = dis_loss(dis, real=real.to(device), fake=fake.to(device))

                    #
                    history['prob_true'][layer].append(dis(real.to(device)).detach().sigmoid().mean().cpu().numpy())
                    history['prob_fake'][layer].append(dis(fake.to(device)).detach().sigmoid().mean().cpu().numpy())

                value = sum(loss.values())

                # backprop for a discriminator of a layer
                for optimizer in optimizer_discriminator.values():
                    optimizer.zero_grad()

                protect(value).backward()
                for optimizer in optimizer_discriminator.values():
                    optimizer.step()

        logger.warning("model_eval")
        for n, dis in discriminators.items():
            dis.cpu()
        with torch.no_grad():
            # eval model
            model.to(device)
            model.eval()

            massiv_logits = []
            massiv_true = []
            for batch_x, batch_y in tqdm.tqdm(task_loaders['val']):
                # calculate CrossEntropyLoss
                # loss_ = gen_loss_task(model,input=batch_x,target = batch_y)
                logits_ = model(batch_x.to(device))

                y_pred_ = logits_.max(1)[1].detach().cpu().numpy()

                val_acc += np.mean(batch_y.cpu().numpy() == y_pred_)

                # calculate predictions in a validation
                val_loss += np.sum(torch.nn.CrossEntropyLoss()(logits_, batch_y.to(device).long()).detach().cpu().numpy())
                del logits_
                del batch_x

                massiv_logits.extend(y_pred_)
                massiv_true.extend([x.item() for x in batch_y.cpu()])

            model.cpu()
            val_acc = balanced_accuracy_score(massiv_true, massiv_logits)
            #  normalizing of loss and accuracy
            val_loss /= len(task_loaders['val'])
            # val_acc /= len(task_loaders['val'])

            history['loss']['val'].append(val_loss)
            history['acc']['val'].append(val_acc)
            print(val_acc)

        logger.warning("finish_eval")
        if epoch > epoch_unfreezing:
            ## feedback for the generator's objective hyperparameters
            out = {}  # inflate, keep, or deflate the weight in the generator's objective
            for layer, value in beta_dictionary.items():
                p_real, p_fake = history['prob_true'][layer][-1], history['prob_fake'][layer][-1]
                if p_fake < 0.2:
                    value = value * 2.

                elif p_fake > 0.45:
                    value = value / 2.

                out[layer] = max(1e-3, min(1e3, value))

                betas_history[layer].append(out[layer])

            beta_dictionary = out

        #clear_output()
        """
        # print results after each epoch
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss (in-iteration): \t{:.6f}".format(train_loss))
        print("  validation loss (in-iteration): \t{:.6f}".format(val_loss))
        print("  training accuracy: \t\t\t{:.2f} %".format(train_acc * 100))
        print("  validation accuracy: \t\t\t{:.2f} %".format(val_acc * 100))

        plot_learning_curves(history)
        """
    return model, history