import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNet(nn.Module):
    '''
    Haitong lee의 exploring knowledge distillation of deep neural networks for efficient hardware solutions 중
    MLP 부분을 참고하였음을 밝힙니다..
    '''
    def __init__(self, config):
        super(LinearNet, self).__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(in_features=28 * 28, out_features=config.hidden_size)
        self.l2 = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.out = nn.Linear(in_features=config.hidden_size, out_features=10)

        self.bn1 = nn.BatchNorm1d(config.hidden_size)
        self.bn2 = nn.BatchNorm1d(config.hidden_size)

        self.dropout1 = nn.Dropout(p=config.dropout_p)
        self.dropout2 = nn.Dropout(p=config.dropout_p)

        self.T = config.temperature
        self.alpha = config.alpha


    def __call__(self, x):
        logit = self.forward(x)
        return logit

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout1(F.relu(self.bn1(self.l1(x))))
        x = self.dropout2(F.relu(self.bn2(self.l2(x))))
        logit = self.out(x)
        return logit

    def loss_fn(self, logit, label):
        '''
        :param logit: 모델의 softmax 입력 전 output
        :param label: ground truth
        :return: 모델의 logit에 대한 교차엔트로피 손실
        '''
        return nn.CrossEntropyLoss()(logit, label)

    def kd_loss_fn(self, student_logit, teacher_logit, label):
        '''
        :param student_logit: small model의 정규화 전 output
        :param teacher_logit: big model의 정규화 전 output
        :param label: groud truth
        :return 정보 증류 손실
        '''
        kd_loss = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(student_logit / self.T, dim=1),
            F.softmax(teacher_logit / self.T, dim=1)) * (self.alpha * self.T * self.T)\
                  + F.cross_entropy(student_logit, label) * (1. - self.alpha)
        return kd_loss

    def accuracy(self, logit, label):
        # |logit| = (bs, n_class)
        # |label| = (bs, 1)
        pred = torch.argmax(logit, dim=1)
        return sum(pred == label) / label.size(0)