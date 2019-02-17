import numpy as np
import torch


class RunningMeanStd(torch.Tensor):
    def __init__(self, epsilon=1e-4, shape=(1)):
        self.mean = torch.zeros(shape)
        self.max = torch.FloatTensor([-np.inf])
        self.var = torch.ones(shape)
        self.count = torch.Tensor([epsilon])
        self.epsilon = torch.Tensor([epsilon])

    def update(self, x):
        if x.size(0) > 1:
            batch_std = torch.std(x, dim=0)
        else:
            batch_std = self.epsilon

        batch_mean, batch_count = torch.mean(x, dim=0), x.size(0)
        batch_var = torch.pow(batch_std, 2)
        self.update_from_moments(batch_mean, batch_var, batch_count)

        self.max = max(x.max().cpu(), self.max)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


if __name__ == "__main__":
    rnm = RunningMeanStd()
    rnm.update(torch.FloatTensor([2]))
    print(rnm.mean, rnm.var, rnm.count)
    rnm.update(torch.FloatTensor([10]))
    print(rnm.mean, rnm.var, rnm.count)
