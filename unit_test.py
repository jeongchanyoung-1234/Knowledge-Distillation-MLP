import argparse
import unittest

import torch
from dataset.data_loader import MNISTDataLoader
from model.linear_net import LinearNet
from trainer import KDTrainer

class CustomTest(unittest.TestCase):

    def setUp(self):
        p = {'batch_size': 128,
             'hidden_size': 100,
             'dropout_p': .5,
             'device': torch.device('cpu'),
             'temperature': 10,
             'lr': 1e-2,
             'epoch': 3,
             'save_model': False}
        self.config = argparse.Namespace(**p)
        self.train_dataloader, self.test_dataloader = MNISTDataLoader(self.config).get_dataloaders(root='./dataset')
        self.model = LinearNet(self.config)

    def test_data_loader(self):
        print('Data loaded: {}'.format(next(iter(self.test_dataloader))[0].size()))

    def test_model(self):
        input = torch.randint(0, 10, (128, 28, 28)).float()
        logit = self.model.forward(input)
        print('forwarded: {}'.format(logit.size()))

    def test_trainer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        trainer = KDTrainer(self.config, self.model, optimizer)
        trainer.train(self.train_dataloader, self.test_dataloader)

if __name__ == '__main__':
    unittest.main()