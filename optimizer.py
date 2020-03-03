from typing import List, Tuple
from torch import optim


class Optimizer:

    def optimizer(self, network, data, data_, batch: int, epoch: int,
                  learning_rate: float, weight_decay: float):

        for epoch_range in range(epoch):

            for batch_range in range(batch):
                pred_x = net(x)
                opt = optim.Adam(net.parameters(), lr=1e-5, weight_decay=0)
                net_loss = net.criterion_l1_loss(pred_x, real_x)
                net_loss.backward()
                opt.step()

                if i % 100 == 0:
                    print(float(net_loss.data))

        # epoch = 1000
        # batch = 10
        #
        # nn.Linear(50, 300)
        # nn.ReLU()
        # nn.Linear(300, 1000)
        # nn.ReLU()
        # nn.Linear(1000, 1000)
        # nn.ReLU()
        # nn.Linear(1000, 50)
        # nn.Sigmoid()

    def create_tuple(self):
        pass

    def shuffle(self):
        pass
