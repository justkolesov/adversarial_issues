from torchvision import transforms, datasets
from data.stanford_dogs_data import dogs
import configs



def load_datasets(name_of_dataset: str):
    """

    :param name_of_dataset: name of  dataset, that will be used for training
    :return: train_resize_data, test_resize_data, classes
    """

    if name_of_dataset == 'stanford_dogs':

        # resize the source data for train and test
        input_transforms = transforms.Compose([
            transforms.Resize((256, 256))])

        train_dataset = dogs(root=configs.configs_path,
                             train=True,
                             cropped=False,
                             transform=input_transforms,
                             download=True)

        test_dataset = dogs(root=configs.configs_path,
                            train=False,
                            cropped=False,
                            transform=input_transforms,
                            download=True)

        classes = train_dataset.classes

        print("Training set stats:")
        train_dataset.stats()
        print("Testing set stats:")
        test_dataset.stats()
    else:
        raise NotImplementedError

    return train_dataset, test_dataset, classes
