import torchvision
from torch.utils.data import DataLoader


class MNISTDataLoader:
    def __init__(self, config):
        self.batch_size = config.batch_size

    def get_dataloaders(self, root:str='.'):
        '''
        dataloader를 불러온다.
        검증용은 따로 나누지 않음.
        params
        root:str train, test폴더의 상위폴더
        return train_dataloader, test_dataloader
        '''
        train_data = torchvision.datasets.MNIST(root=root + '/train', train=True, download=True,
                                        transform=torchvision.transforms.ToTensor())
        test_data = torchvision.datasets.MNIST(root=root + '/test', train=False, download=True,
                                               transform=torchvision.transforms.ToTensor())

        train_dataloader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True, num_workers=4)
        test_dataloader = DataLoader(dataset=test_data, batch_size=self.batch_size, shuffle=False, num_workers=4)

        return train_dataloader, test_dataloader



