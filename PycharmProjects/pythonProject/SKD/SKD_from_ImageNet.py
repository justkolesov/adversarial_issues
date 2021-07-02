import torch

def get_filters(weights):
    li = []
    klyuchi = []
    kq = [3, 4, 6, 3]
    for key in weights["state_dict"].keys():
        if key == "module.encoder_q.conv1.weight":
            li.append(weights["state_dict"][key])
            klyuchi.append(key)
            continue
        if "conv" in key:
            for layer in range(1, 5):
                for s in range(kq[layer - 1]):
                    for k in range(1, 4):
                        term = "module.encoder_q.layer" + str(layer) + "." + str(s) + ".conv" + str(k) + ".weight"
                        if key == term:
                            li.append(weights["state_dict"][key])
                            klyuchi.append(key)

    for  key in weights["state_dict"].keys():
      for s in range(1,5):
        term = "module.encoder_q.layer" + str(s) + ".0.downsample.0.weight"
        if key == term:
          li.append(weights["state_dict"][key])
          klyuchi.append(key)

    filters_list = [li[idx].reshape(-1, li[idx].shape[-1:].numel(), li[idx].shape[-1:].numel()).detach().cpu() for idx
                    in range(len(li))]

    name_list_without_weight = klyuchi
    name_list_without_weight[:49] = [name[17:] for name in name_list_without_weight[:49]]
    name_list_without_weight[-4:] = [name[17:] for name in name_list_without_weight[-4:]]
    name_list_without_weight = [name[:-7] for name in name_list_without_weight]

    SKD_convolutional_weights = {name: filters for name, filters in zip(name_list_without_weight, \
                                                                        filters_list)}

    return SKD_convolutional_weights