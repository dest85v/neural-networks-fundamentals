import torchvision.datasets

class MNISTSimpleDataset:
    def __init__(self, train=True):
        ## Load MNIST dataset here
        ## YOUR CODE HERE
        dataset = torchvision.datasets.MNIST(
            root='~/.cache/mnist',
            train=train,
            download=True
        )
        self.X = dataset.data.float()
        self.y = dataset.targets.long()


    def __len__(self):
        res = 0
        ## Return number of items that is there in the dataset
        ## YOUR CODE HERE
        res = len(self.X)
        return res


    def __getitem__(self, index):
        sample = {}

        ## Return a sample of the dataset that corresponds to the input index
        ## YOUR CODE HERE
        image = self.X[index] / 255.0 * 2.0 - 1.0
        label = self.y[index]
        sample['image'] = image
        sample['label'] = label
        
        return sample 
