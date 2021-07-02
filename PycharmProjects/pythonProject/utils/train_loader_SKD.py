import torch
import configs


class LiveConv2dSliceInfiniteSampler:

    def __init__(self, module, batch_size=None):
        assert isinstance(module, torch.nn.Conv2d)
        self.module = module
        self.batch_size = batch_size
        self.shape = self.module.weight.shape[-2:]
        # self.shape is equal to shape of filter in a layer
        # for instance, self.shape = torch.tensor([3,3])
        # if kernel_size of the layer equals 3

    def __str__(self):
        return f"module is {self.module} and batch_size is {self.batch_size}"

    # namely it is used for object of class
    def __repr__(self):
        text = f" module = {self.module} , batch_size = {self.batch_size}"
        return type(self).__name__ + "(" + text + ")"

    def __len__(self):
        return self.module.weight.shape[:-2].numel()

    def __iter__(self):
        n_batches = (self.__len__() + self.batch_size - 1) // (self.batch_size)
        while True:
            sequence = torch.randperm(self.__len__(), device=self.module.weight.device)  # the second parameter is cuda
            # before for to avoid intersections of batches
            for i in range(n_batches):
                view = self.module.weight.view(-1, *self.shape)
                yield view[sequence[i * self.batch_size: (i + 1) * self.batch_size]]


def create_SKD(SKD_convolutional_weights: dict, model):
    """

    :param SKD_convolutional_weights: the dict {"name_layer", tensor_weights}
    :param BATCH_CONVO_LIST: batch of each discriminator of layers
    :param model: ResNet torchvision model
    :return:
    """

    batch_convo_filters = {name: batch for name, batch in zip(SKD_convolutional_weights.keys(), configs.batch_convo_list)}

    live_sampler_gen = {name: LiveConv2dSliceInfiniteSampler(module, batch_size=batch_convo_filters[name])
                        for name, module in model.named_modules() if isinstance(module, torch.nn.Conv2d)}

    ker_iters_gen = {layer_name: iter(smplr) for layer_name, smplr in live_sampler_gen.items()}



    batch_size_SKD = {name: batch for name, batch in zip(SKD_convolutional_weights.keys(), configs.batch_skd)}

    train_loaders_SKD = {
        name: torch.utils.data.DataLoader(dataset=SKD_convolutional_weights[name], batch_size=batch_size, shuffle=True) \
        for name, batch_size in batch_size_SKD.items()}

    return train_loaders_SKD, ker_iters_gen