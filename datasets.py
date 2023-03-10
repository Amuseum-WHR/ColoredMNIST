import os
import torch
from PIL import ImageFile
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST

ImageFile.LOAD_TRUNCATED_IMAGES = True


class MultipleEnvironmentMNIST:
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        # super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

class ColoredMNIST(MultipleEnvironmentMNIST):

    def __init__(self, root, env = [0.1]):
        super(ColoredMNIST, self).__init__(root, env,
                                         self.color_dataset, (2, 28, 28,), 10)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # Assign a binary label based on the digit
        label = (labels < 5).float()
        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(label,
                                 self.torch_bernoulli_(environment,
                                                       len(label)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()