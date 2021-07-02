import torch
import torchvision

def ResNet50_filters_from_Imagenet_init(model : torchvision.models, num_classes : int) -> torchvision.models:
    """

    :param model: ResNet50 model with random initial filters
    :param num_classes: num classes of train data
    :return: ResNet50 model with self-supervised filters
    """

    try:
        model.fc = torch.nn.Sequential(torch.nn.Linear(2048, 2048),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(2048, 128))
    except AttributeError as err:
        raise AssertionError

    state = torch.load("./moco_v2_800ep_pretrain.pth.tar")['state_dict']
    state = {".".join(k.split(".")[2:]):v for k,v in state.items()}

    model.load_state_dict(state, strict=False)
    model.fc = torch.nn.Linear(2048, num_classes)

    return model