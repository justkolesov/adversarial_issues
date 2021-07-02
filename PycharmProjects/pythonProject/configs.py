import os

root = os.path.expanduser(".")
configs_path = os.path.join(root, 'DATASETS', 'IMAGE')

"  Mean and  Test from imagenet "
train_test_MEAN_ImageNet = [0.485,0.456,0.406]
train_test_STD_ImageNet = [0.229,0.224,0.225]

BETA_LIST = [1e-3]*53

"the structure of advesarial discriminators"
hidden_discr = [100, 100]
learning_rate_discriminators = [1e-3]*53 # ResNet50


"learning rate for discriminators"


" Batch of discriminators for convolutional layers"
batch_convo_list =  [192]*53

batch_skd =[192]*53