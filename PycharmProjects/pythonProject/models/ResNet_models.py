import torchvision

def ResNet50(pretrained=False):
    """

    :param pretrained:
    :return: ResNe50 model
    """
    return torchvision.models.resnet50(pretrained=pretrained)