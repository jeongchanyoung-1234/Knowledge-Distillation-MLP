import argparse

import torch
from dataset.data_loader import MNISTDataLoader
from model.linear_net import LinearNet
from trainer import KDTrainer


def define_argparser():
    '''아래 기본값은 teacher model 학습용이므로
    증류 시 히든사이즈를 800으로, 학습율을 1e-1로 줄이시기 바랍니다.
    batch_size: '''
    p = argparse.ArgumentParser()
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--hidden_size', type=int, default=1200)
    p.add_argument('--dropout_p', type=float, default=.8)
    p.add_argument('--alpha', type=float, default=.1)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--temperature', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--weight_decay', type=float, default=0)
    p.add_argument('--epoch', type=int, default=10)
    p.add_argument('--teacher_model_pth', type=str, default=None)
    p.add_argument('--save_model', action='store_true')

    config = p.parse_args()
    config.device = torch.device(config.device)
    return config


def main(config):
    model = LinearNet(config)
    train_dataset, test_dataset = MNISTDataLoader(config).get_dataloaders(root='./dataset')
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=.9, weight_decay=config.weight_decay)

    trainer = KDTrainer(config, model, optimizer)
    trainer.train(train_dataset, test_dataset)

if __name__ == '__main__':
    config = define_argparser()
    main(config)