"""
    We try to determine if it is harder for a NN to learn from
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import os

from typing import Tuple, List

from torch.utils.tensorboard import SummaryWriter

MNIST_SIZE = 5


def transform_and_save_data_set(size: int, num_workers: int = 8, folder: str = "../data",
                                dataset_name="CIFAR10"):
    norm = (0.1307,), (0.3081,)
    size_img = (size, size)
    mnist_path = f"{folder}/{dataset_name}_{size}_dataset"

    if os.path.isfile(mnist_path):
        return mnist_path

    data_set_class = getattr(datasets, dataset_name)
    train_loader = torch.utils.data.DataLoader(
        data_set_class(folder, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(size_img),
                           transforms.ToTensor(),
                           transforms.Normalize(*norm)
                       ])),
        batch_size=60000, shuffle=False, num_workers=num_workers)

    data, targets = next(iter(train_loader))
    sorted_train_targets, idxs = targets.sort()
    sorted_train_data = data[idxs]

    test_loader = torch.utils.data.DataLoader(
        data_set_class(folder, train=False, transform=transforms.Compose([
            transforms.Resize(size_img),
            transforms.ToTensor(),
            transforms.Normalize(*norm)
        ])),
        batch_size=10000, shuffle=True, num_workers=num_workers)

    data, targets = next(iter(test_loader))
    sorted_test_targets, idxs = targets.sort()
    sorted_test_data = data[idxs]

    torch.save({
        "test": (sorted_test_data, sorted_test_targets),
        "train": (sorted_train_data, sorted_train_targets),
    }, mnist_path)
    return mnist_path


class DataSetMem:
    def __init__(self, size: int = 5, device: str = "cpu", train: bool = True, data_name="CIFAR10"):
        data_path = transform_and_save_data_set(size=size, dataset_name=data_name)

        self._data_name = data_name
        self._train = train
        self._all_data = data = torch.load(data_path)

        for ttt in ["train", "test"]:
            data[ttt] = list(data[ttt])
            data[ttt][0] = data[ttt][0].flatten(1).to(device)
            data[ttt][1] = data[ttt][1].to(device)

        crt_type = "train" if train else "test"
        self._data = data[crt_type][0]
        self._targets = data[crt_type][1]

        self._size = size ** 2
        self._num_classes = 10
        self._permute = torch.arange(self._size)

    def size(self):
        return self._size

    def change_data(self, train: bool):
        crt_type = "train" if train else "test"
        self._train = train
        self._data = self._all_data[crt_type][0]
        self._data = self._data[:, self._permute]
        self._targets = self._all_data[crt_type][1]

    def change_permute(self, permute: torch.Tensor):
        self._permute = permute
        self.change_data(self._train)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx], self._targets[idx]


class TaskData:
    def __init__(self, data_set: DataSetMem, num_tasks: int, batch_size: int,
                 device: str = "cpu", shuffle: bool = True):
        assert num_tasks > 0, "Must have min 1 task"
        self.data_set = data_set
        self._num_tasks = num_tasks
        self._item_size = data_set.size()
        self._size = data_set.size() * 1
        self._num_classes = data_set._num_classes
        self._crt_task = 0
        self._batch_size = batch_size
        self._device = device
        self._shuffle = shuffle
        self._permute = torch.arange(data_set.size())

    def size(self):
        return self._size

    @property
    def dataset(self):
        return self.data_set

    def __len__(self):
        return len(self.data_set)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_tasks(self):
        return self._num_tasks

    @property
    def task(self):
        return self._crt_task

    @task.setter
    def task(self, value: int):
        assert 0 <= value < self._num_tasks
        self._set_new_task(value)
        self._crt_task = value

    def next_task(self):
        new_task = (self._crt_task + 1) % self._num_tasks
        self.task = new_task

    def _set_new_task(self, new_task: int):
        pass

    def get_data(self):
        return self._get_data()

    def _get_data(self):
        data = self.data_set
        batch_size = self._batch_size
        idx_s = torch.randperm(len(data)) if self._shuffle else torch.arange(len(data))

        for idx in range(0, idx_s.size(0) // batch_size * batch_size, batch_size):
            fetch = idx_s[idx: idx+batch_size]
            yield data[fetch]


class SeqTasksSameTarget(TaskData):
    def __init__(self, data_set: DataSetMem, num_tasks: int, batch_size: int,
                 device: str = "cpu"):
        super(SeqTasksSameTarget, self).__init__(data_set, num_tasks, batch_size, device)
        self._size = self.data_set.size() * num_tasks

    def __iter__(self):
        return self.get_data()

    def get_data(self):
        task = self._crt_task
        item_size = self._item_size
        in_pos = item_size * task
        fin_pos = item_size * (task + 1)
        batch_size = self._batch_size
        size = self.size()
        device = self._device

        for i, (data, target) in enumerate(self._get_data()):
            data_storage = torch.zeros(batch_size, size, device=device)
            data_storage[:, in_pos: fin_pos].copy_(data)
            yield data_storage.detach(), target


class SeqTasksSameTargetNoise(TaskData):
    def __init__(self, data_set: DataSetMem, num_tasks: int, batch_size: int,
                 device: str = "cpu"):
        super(SeqTasksSameTargetNoise, self).__init__(data_set, num_tasks, batch_size, device)
        self._size = self.data_set.size() * num_tasks

    def __iter__(self):
        return self.get_data()

    def get_data(self):
        task = self._crt_task
        num_tasks = self._num_tasks
        item_size = self._item_size
        in_pos = item_size * task
        fin_pos = item_size * (task + 1)
        batch_size = self._batch_size
        size = self.size()
        device = self._device

        data_set = self.data_set

        for i, (data, target) in enumerate(self._get_data()):
            batch = data_set[torch.randperm(len(data_set))[:batch_size*num_tasks]][0]
            data_storage = batch.view(batch_size, -1)
            data_storage[:, in_pos: fin_pos].copy_(data)
            yield data_storage.detach(), target

    # def _set_new_task(self, new_task: int):
    #     self._permute = torch.randperm(self.data_set.size())
    #     self.data_set.change_permute(self._permute)


class MultiDataset(TaskData):
    def __init__(self, data_sets: List[DataSetMem], num_tasks: int, batch_size: int,
                 device: str = "cpu"):
        self._all_data_sets = data_sets
        super(MultiDataset, self).__init__(data_sets[0], num_tasks, batch_size, device)
        self._size = self.data_set.size() * num_tasks

    def __iter__(self):
        return self.get_data()

    def get_data(self):
        task = self._crt_task
        num_tasks = self._num_tasks
        item_size = self._item_size
        in_pos = item_size * task
        fin_pos = item_size * (task + 1)
        batch_size = self._batch_size
        size = self.size()
        device = self._device

        data_set = self.data_set

        print(self.data_set._data_name, in_pos, fin_pos)
        for i, (data, target) in enumerate(self._get_data()):
            data_storage = torch.zeros(batch_size, size, device=device)
            data_storage[:, in_pos: fin_pos].copy_(data)
            yield data_storage.detach(), target

    def _set_new_task(self, new_task: int):
        self.data_set = self._all_data_sets[new_task]


class Net(nn.Module):
    def __init__(self, in_size: int, out_size: int, num_hidden_layers: int, hidden_size: int = 256):
        super(Net, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU()
        )

        self.has_hidden = num_hidden_layers > 0
        self.hfc = nn.Sequential()
        for i in range(num_hidden_layers):
            self.hfc.add_module(f"h{i}_fc", nn.Linear(hidden_size, hidden_size))
            self.hfc.add_module(f"h{i}_act", nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, out_size),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.flatten(1)
        x = self.fc1(x)
        if self.has_hidden:
            x = self.hfc(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(start_i, args, model, device, train_loader, optimizer, epoch, writer, run=0,
          plt_name="Train"):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} | task: {train_loader.task} | Loss: {loss.item():.6f}')
            writer.add_scalars(f'{plt_name}/multi', {f'train_loss_{run}': loss.item()},
                               start_i + batch_idx)
    return start_i + batch_idx


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        acc))
    return test_loss, acc


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    writer = SummaryWriter()

    img_size = 8
    num_tasks = 3
    num_layers = 2
    hidden_size = 256

    base_dataset = DataSetMem(size=img_size, device=device, train=True, data_name="MNIST")
    base_dataset2 = DataSetMem(size=img_size, device=device, train=True, data_name="FashionMNIST")
    base_dataset3 = DataSetMem(size=img_size, device=device, train=True, data_name="CIFAR10")

    # train_loader = SeqTasksSameTarget(base_dataset, num_tasks, args.batch_size, device=device)

    for base_dataset, plt_name in [
        ([base_dataset, base_dataset2, base_dataset3], "Train_MFC"),
        ([base_dataset2, base_dataset3, base_dataset], "Train_FCM"),
        ([base_dataset3, base_dataset, base_dataset2], "Train_CMF")
    ]:
    # base_dataset = [base_dataset, base_dataset2, base_dataset3]
    # plt_name = "Train_MFC"
        for run in range(5):
            torch.manual_seed(args.seed + run)

            train_loader = MultiDataset(base_dataset, num_tasks, args.batch_size, device=device)

            model = Net(train_loader.size(), train_loader.num_classes, num_layers,
                        hidden_size=hidden_size).to(device)

            # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
            optimizer = optim.Adadelta(model.parameters(), lr=0.003, eps=1.e-5)
            # optimizer = optim.Adam(model.parameters(), lr=0.0001)
            # optimizer = optim.RMSprop(model.parameters(), lr=0.00003, eps=1.e-6, alpha=0.99)
            # optimizer = optim.RMSprop(model.parameters(), lr=0.0001, eps=1.e-5, alpha=0.99)

            scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
            start_idx = 0
            for task_id in range(train_loader.num_tasks):
                for epoch in range(1, args.epochs + 1):
                    train_loader.data_set.change_data(True)
                    start_idx = train(start_idx, args, model, device, train_loader, optimizer, epoch,
                                      writer, run=run, plt_name=plt_name)

                    train_loader.data_set.change_data(False)
                    loss, acc = test(args, model, device, train_loader)
                    writer.add_scalars(f'{plt_name}/eval_acc', {f"eval_acc_{run}": acc}, start_idx)
                    writer.add_scalars(f'{plt_name}/multi', {f"eval_loss_{run}": loss}, start_idx)

                    # scheduler.step()
                train_loader.next_task()

            if args.save_model:
                torch.save(model.state_dict(), "mnist_cnn.pt")


def analysis_logs():
    import glob
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import re
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')

    log_files = glob.glob("runs/Feb08_19-53-42_andrei/*/events.*", recursive=True)
    print(len(log_files))
    datas = []
    data = {}
    for fl in log_files:
        path = os.path.basename(os.path.dirname(fl))
        data[path] = dict()
        event_acc = EventAccumulator(fl)
        event_acc.Reload()
        scalars = event_acc.Tags()["scalars"]
        for sc in scalars:
            data[path][sc] = event_acc.Scalars(sc)
            run = int(re.search(r'\d+', path[::-1]).group()[::-1])
            for d_point in data[path][sc]:
                datas.append([path, sc, run, d_point.step, d_point.value])

    df = pd.DataFrame(datas, columns=["path", "log", "run", "step", "value"])

    eval_dfs = df[df.log.apply(lambda x: "eval_acc" in x)]

    fig1, ax1 = plt.subplots()
    for exp, exp_df in eval_dfs.groupby("log"):
        plt_data = exp_df.groupby("step").mean()
        plt_data["value"].plot(label=exp)
    ax1.legend()
    plt.title("mean")
    plt.show()
if __name__ == '__main__':
    main()


