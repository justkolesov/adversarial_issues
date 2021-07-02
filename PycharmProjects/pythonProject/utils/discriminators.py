import torch
import torchvision
import configs

class Discriminator(torch.nn.Module):
    def __new__(cls, input_dim, hidden):
        assert isinstance(hidden, list)

        model = [torch.nn.Flatten()]
        #model.pop()


        hidden_layers = [input_dim] + hidden + [1]
        for in_features, out_features in zip(hidden_layers, hidden_layers[1:]):
            model.extend([
                torch.nn.Linear(in_features, out_features),
                torch.nn.ReLU()
            ])
        model.pop()

        return torch.nn.Sequential(*model)

def gen_loss_dis(discriminator, *, fake):
    return -discriminator(fake).mean()

def dis_loss(discriminator,  real, fake, ell=torch.nn.BCEWithLogitsLoss()):
    out_real = discriminator(real)
    out_fake = discriminator(fake.detach())

    loss_real = ell(out_real, torch.full_like(out_real, 0.9))
    loss_fake = ell(out_fake, torch.full_like(out_fake, 0.))
    return loss_real * 0.5 + loss_fake * 0.5



def create_discriminators_resnet50(model: torchvision.models, device:str) -> dict:
    """

    :param model:
    :param device:
    :return:
    """
    discriminators = {
        name: Discriminator(module.weight.shape[-2:].numel(), configs.hidden_discr).to(device)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Conv2d)
    }

    a = discriminators['layer1.0.downsample.0']
    b = discriminators['layer2.0.downsample.0']
    c = discriminators['layer3.0.downsample.0']
    d = discriminators['layer4.0.downsample.0']
    del discriminators['layer1.0.downsample.0']
    del discriminators['layer2.0.downsample.0']
    del discriminators['layer3.0.downsample.0']
    del discriminators['layer4.0.downsample.0']

    discriminators['layer1.0.downsample.0'] = a
    discriminators['layer2.0.downsample.0'] = b
    discriminators['layer3.0.downsample.0'] = c
    discriminators['layer4.0.downsample.0'] = d


    return discriminators