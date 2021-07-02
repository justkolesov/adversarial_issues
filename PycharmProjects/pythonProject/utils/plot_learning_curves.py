import matplotlib.pyplot as plt

def plot_learning_curves(history, name_list_without_weight, batch_size_SKD, num_layers:int):
    fig = plt.figure(figsize=(50, 20))

    plt.subplot(1, 2, 1)
    plt.plot(history['loss']['train'], color='indigo', label='full loss train')
    plt.plot(history['task_loss']['train'], color='tomato', label='task loss')
    plt.plot(history['pseudo_gen_loss']['train'], color='gold', label='pseudo gen loss')
    plt.plot(history['loss']['val'], color='limegreen', label='validation loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['acc']['train'], color='midnightblue', label='train accuracy')
    plt.plot(history['acc']['val'], color='firebrick', label='validation accuracy')
    plt.legend()

    fig, ax = plt.subplots( num_layers, 1, figsize=(20, 200))
    ax = ax.flatten()
    for idx, layers_, bt in zip(range(num_layers), name_list_without_weight, list(batch_size_SKD.values())):
        ax[idx].plot(history['prob_true'][layers_], color='purple', label='SKD filters ' + str(idx + 1) + ' layer')
        ax[idx].plot(history['prob_fake'][layers_], color='salmon', label='fitness filters' + str(idx + 1) + ' layer')
        ax[idx].axhline(0.5, color='blue')
        ax[idx].set_ylim([0, 1])
        ax[idx].set_ylabel('probability')
        ax[idx].set_xlabel('num iterations')
        ax[idx].set_title('500 samples per train : Batch ' + str(bt))
        ax[idx].legend()

    plt.show()

# auxiliary function for calculation
# If loss will be nan or inf
# then there will be exception
def protect(scalar):
    """Raise if a scalar is either NaN, or Inf."""
    if not isfinite(float(scalar)):
        raise FloatingPointError

    return scalar

def plot_plot(a,b):
    plt.plot(a, color ='blue', label='init:self_fltrs + adv_train')
    plt.plot(b, color = 'darkgreen', label = 'init:self_flts + ce_train')
    plt.title("ImageNet -> Stanford dogs : 5 objects per class")
    plt.xlabel("num epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig("./plots/plot_ImageNet_Stanford_5_obj_per_train")

