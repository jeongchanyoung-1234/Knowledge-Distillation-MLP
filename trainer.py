import time
import pprint

import torch


class KDTrainer:
    def __init__(self,
                 config,
                 model,
                 optimizer):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        if config.teacher_model_pth is None:
            self.teacher_model = None
        else:
            self.teacher_model = torch.load(config.teacher_model_pth)


    def train(self, train_dataloader, valid_dataloader):
        self.model.to(self.config.device)
        self.model.train()
        if self.teacher_model is not None :
            self.teacher_model.to(self.config.device)
            self.teacher_model.eval()

        print('[CONFIG]')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(self.config))
        start = time.time()
        for epoch in range(self.config.epoch):
            total_train_loss = 0
            total_train_acc = 0
            train_cnt = 0
            for batch_x, batch_y in train_dataloader:
                batch_x, batch_y = batch_x.to(self.config.device), batch_y.to(self.config.device)

                logit = self.model.forward(batch_x)
                if self.teacher_model is not None:
                    teacher_logit = self.teacher_model.forward(batch_x)
                    train_loss = self.model.kd_loss_fn(logit, teacher_logit, batch_y)
                else:
                    train_loss = self.model.loss_fn(logit, batch_y)
                train_acc = self.model.accuracy(logit, batch_y)

                train_cnt += 1
                total_train_loss += train_loss
                total_train_acc += train_acc

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

            if valid_dataloader is not None:
                self.model.eval()
                total_valid_loss = 0
                total_valid_acc = 0
                valid_cnt = 0
                for batch_x, batch_y in valid_dataloader:
                    batch_x, batch_y = batch_x.to(self.config.device), batch_y.to(self.config.device)
                    logit = self.model.forward(batch_x)
                    valid_loss = self.model.loss_fn(logit, batch_y)
                    valid_acc = self.model.accuracy(logit, batch_y)

                    valid_cnt += 1
                    total_valid_loss += valid_loss
                    total_valid_acc += valid_acc

                print("(EPOCH {}/{}) train_loss={:.4f} valid_loss={:.4f} train_accuracy={:.2f}% valid_accuracy={:.2f}% time={:.4f}(sec)".format(
                    epoch + 1,
                    self.config.epoch,
                    total_train_loss.detach() / train_cnt,
                    total_valid_loss.detach() / valid_cnt,
                    total_train_acc.detach() / train_cnt * 100,
                    total_valid_acc.detach() / valid_cnt * 100,
                    time.time() - start
                ))
            else:
                print(
                    "(EPOCH {}/{}) train_loss={:.} train_accuracy={} time={}(sec)".format(
                        epoch + 1,
                        self.config.epoch,
                        train_loss.detach(),
                        train_acc.detach(),
                        time.time() - start
                    ))

        if self.config.save_model:
            name = 'linearnet.acc{}.batch{}.hidden{}'.format(
                int(train_acc * 100),
                self.config.batch_size,
                self.config.hidden_size,
            )
            self.save_model(name)

    def save_model(self, name, path='./model/save/'):
        torch.save(self.model, path + name + '.pth')

    def load_model(self, name, path='./model/save/'):
        self.model = torch.load(path + name + 'pth')