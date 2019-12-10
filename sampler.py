import torch
import torch.utils.data
import torchvision

#taken from https://github.com/ufoym/imbalanced-dataset-sampler/

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw (batch size)
        Use this sampler to select data equally from all the labels
        please note: shuffle should be false when using this sampler
        example:
        train_dataset = ClasswiseDataset(train_dir, transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(brightness=0, contrast=0.1, saturation=0, hue=0),
            transforms.RandomRotation(20,expand=True)]),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
     sampler=ImbalancedDatasetSampler(train_dataset,num_samples=64)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=sampler, shuffle=False, batch_size=64, 
                                               num_workers=2, pin_memory=True) 
    """

    def __init__(self, dataset, indices=None, num_samples=None):
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.labels[idx]
#         dataset_type = type(dataset)
#         if dataset_type is torchvision.datasets.MNIST:
#             return dataset.train_labels[idx].item()
#         elif dataset_type is torchvision.datasets.ImageFolder:
#             return dataset.imgs[idx][1]
#         else:
#             raise NotImplementedError
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
