import torch
from torchvision import transforms
import configs


train_transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = configs.train_test_MEAN_ImageNet,
                             std = configs.train_test_STD_ImageNet)
    ])

test_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = configs.train_test_MEAN_ImageNet,
                             std = configs.train_test_STD_ImageNet)
    ])

def collate_train(batch : int):
    """

    :param batch: batch of train data
    :return: modified batch of train data
    """
    tensor_img = torch.randn(1,3,224,224)
    tensor_lbls = []
    for x, y in batch:
        tensor_img = torch.cat([tensor_img,train_transform(x).reshape(1,3,224,224)])
        tensor_lbls.append(y)
    return (tensor_img[1:], torch.tensor(tensor_lbls))



def collate_test(batch : int):
    """

    :param batch: batch of test data
    :return: modified batch of test data
    """
    tensor_img = torch.randn(1,3,224,224)
    tensor_lbls = []
    for x, y in batch:
        tensor_img = torch.cat([tensor_img,test_transform(x).reshape(1,3,224,224)])
        tensor_lbls.append(y)
    return (tensor_img[1:], torch.tensor(tensor_lbls))



def task_loaders_function(train_data, test_data, batch_size : int) -> dict:
    """

    :param train_data: small train data
    :param test_data:  full test data
    :param batch_size: batch for training
    :return: dict of dataloaders for train, test and validation
    """
    task_loaders = {
        'train': torch.utils.data.DataLoader(
            train_data, batch_size=batch_size,  collate_fn=collate_train, shuffle=True),
        'val': torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, collate_fn=collate_test, shuffle=False),
        'test': torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, collate_fn=collate_test, shuffle=False)
    }

    return task_loaders