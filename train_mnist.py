from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5, 1)
#         self.conv2 = nn.Conv2d(20, 50, 5, 1)
#         self.fc1 = nn.Linear(4 * 4 * 50, 500)
#         self.fc2 = nn.Linear(500, 10)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 4 * 4 * 50)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)


def get_stats(act):
    return (act == 0).cpu(), (act > 0).cpu()


def apply_mask(x, mask):
    print("Do nothing with mask")
    return x


def apply_mask(x, mask):
    x = x * mask
    return x


def apply_mask(x, mask):
    mask, ch = mask
    sel = ~mask.bool()
    x[:, sel] = ch[sel].expand(x.size(0), sel.sum())
    return x


def apply_mask(x, mask):
    q = torch.rand(mask.size(0)).cuda()
    sel = ~mask.bool()
    mask[sel] += q[sel]
    x = x * mask
    return x


def apply_mask(x, mask):
    t = torch.range(0, x.size(1)-1).long()
    t = t[~(mask.bool())]
    shuffle = t[torch.randperm(t.size(0))]
    x[:, t] = x[:, shuffle]
    return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 20, 5, 1)
        # self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc0 = nn.Linear(784, 500)
        self.fc1 = nn.Linear(500, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x, stats=False, masks=[None, None]):
        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        x0 = x
        l0 = get_stats(x)
        x = F.relu(x)
        r0 = x
        if masks[0] is not None:
            x = apply_mask(x, masks[0])

        x = self.fc1(x)
        x1 = x
        l1 = get_stats(x)
        x = F.relu(x)
        r1 = x
        if masks[1] is not None:
            x = apply_mask(x, masks[1])

        x = self.fc2(x)

        if stats:
            return F.log_softmax(x, dim=1), [l0, l1, x0.cpu(), x1.cpu(), r0.cpu(), r1.cpu()]
        else:
            return F.log_softmax(x, dim=1)


saved_batch = None
saved_batch_clone = None
all_stats = []
test_stats = []
models = {}
save_models = True


def train(args, model, device, train_loader, optimizer, epoch):
    global saved_batch, all_stats, saved_batch_clone, save_models, models
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output, stats = model(data, stats=True)

        if saved_batch is None:
            print("SAVE BATCH")
            saved_batch = (data.clone(), target.clone())
            # saved_batch_clone = (data.clone(), target.clone())
            all_stats.append({
                "stats": stats,
                "loss": loss,
            })

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

            # -- save stats
            output, stats = model(saved_batch[0], stats=True)
            all_stats.append({
                "stats": stats,
                "loss": loss,
            })
            if save_models and batch_idx * len(data) == 58880:
                print("Saved model")
                new_model = Net().to(device)
                new_model.load_state_dict(model.state_dict())
                models[len(all_stats)-1] = new_model

    print(len(all_stats))


def compare_separations(saved_batch, all_stats):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd


    # ====
    stored_data = torch.load("test_rms_default_max96acc")
    saved_batch = stored_data["saved_batch"]
    all_stats = stored_data["all_stats"]
    test_stats = stored_data["test_stats"]
    # ====

    data, target = saved_batch
    data = data.cpu()
    target = target.cpu()

    common0a = []
    common1a = []
    common0aV = dict()
    common1aV = dict()

    common0e = []
    common1e = []
    common0eV = dict()
    common1eV = dict()

    for compare_epoch in range(len(all_stats)-1):
        print(compare_epoch)
        common0 = []
        common1 = []
        common0aV[compare_epoch] = dict()
        common1aV[compare_epoch] = dict()
        common0eV[compare_epoch] = dict()
        common1eV[compare_epoch] = dict()

        common_till_l0_end = None
        common_till_l1_end = None
        idxs = []

        stats_activations_h = ["same_a0_avg", "same_a0_std", "diff_a0_avg", "diff_a0_std",
                               "same_a1_avg", "same_a1_std", "diff_a1_avg", "diff_a1_std"]
        stats_activations = []
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        for epoch in range(compare_epoch + 1, len(all_stats)):
            l0, l1, x0, x1, _, _ = all_stats[epoch]["stats"]
            l0_0, l0_g = l0
            l1_0, l1_g = l1

            common_bits_l1 = l0_g == all_stats[compare_epoch]["stats"][0][1]
            common_bits_l2 = l1_g == all_stats[compare_epoch]["stats"][1][1]

            common_bits_l1 = common_bits_l1.sum(dim=0) >= common_bits_l1.size(0)
            common_bits_l2 = common_bits_l2.sum(dim=0) >= common_bits_l2.size(0)

            # ======================================================================================
            # activation stats
            # a0 = all_stats[epoch]["stats"][2][:, common_bits_l1]
            # a0 = a0.abs()
            #
            # na0 = all_stats[epoch]["stats"][2][:, ~common_bits_l1]
            # na0 = na0.abs()
            #
            # a1 = all_stats[epoch]["stats"][3][:, common_bits_l2]
            # a1 = a1.abs()
            #
            # na1 = all_stats[epoch]["stats"][3][:, ~common_bits_l2]
            # na1 = na1.abs()
            #
            # stats_activations.append([a0.mean().item(), a0.std().item(),
            #                           na0.mean().item(), na0.std().item(),
            #                           a1.mean().item(), a1.std().item(),
            #                           na1.mean().item(), na1.std().item()
            #                           ])
            # ======================================================================================

            if common_till_l0_end is None:
                common_till_l0_end = common_bits_l1
                common_till_l1_end = common_bits_l2
            else:
                common_till_l0_end = common_till_l0_end & common_bits_l1
                common_till_l1_end = common_till_l1_end & common_bits_l2

            common0.append(common_bits_l1.sum().item() / float(common_bits_l1.size(0)))
            common1.append(common_bits_l2.sum().item() / float(common_bits_l2.size(0)))
            idxs.append(epoch)

            common0aV[compare_epoch][epoch] = common_bits_l1
            common1aV[compare_epoch][epoch] = common_bits_l2

            # for label in torch.unique(target).numpy():
            #     print("-"*10,  label)
            #     t = target == label
            #
            #     for name, l in [("l0_g", l0_g), ("l1_g", l1_g)]:
            #         s = l[t]
            #         same = s[1:] == s[0]
            #         common_bits = same.sum(dim=0) == same.size(0)
            #         print(name, label, common_bits.sum().item()/float(common_bits.size(0)), f"(s:{common_bits.size(0)})")

            # f, axes = plt.subplots(4, 1, sharex=True)
            # f.suptitle(f"{epoch}")
            # axes[0].hist(a0.data.numpy().reshape(-1), bins=100)
            # axes[1].hist(na0.data.numpy().reshape(-1), bins=100)
            # axes[2].hist(a1.data.numpy().reshape(-1), bins=100)
            # axes[3].hist(na1.data.numpy().reshape(-1), bins=100)
            # plt.show()
            # plt.waitforbuttonpress()
            # plt.close()

            # hist, bins = np.histogram(na0.data.numpy().reshape(-1), bins=100)
            # xs = (bins[:-1] + bins[1:]) / 2
            #
            # ax.bar(xs, hist, zs=epoch,  alpha=0.8)
        # plt.show()

        # df = pd.DataFrame(stats_activations, columns=stats_activations_h)
        # df.plot()
        # plt.show()
        # plt.waitforbuttonpress()
        # plt.close()

        common0a.append([common0, idxs])
        common1a.append([common1, idxs])
        common0e.append(common_till_l0_end.sum().item() / float(common_till_l0_end.size(0)))
        common1e.append(common_till_l1_end.sum().item() / float(common_till_l1_end.size(0)))

        common0eV[compare_epoch] = common_till_l0_end
        common1eV[compare_epoch] = common_till_l1_end

    common0a = np.array(common0a)
    common1a = np.array(common1a)
    common0e = np.array(common0e)
    common1e = np.array(common1e)
    max_r = len(all_stats)

    loss = [all_stats[i]["loss"] for i in range(max_r)]

    acc = [x["accuracy"] for x in test_stats]
    acc_idx = np.arange(max_r/10, max_r+1, max_r/10)
    f, axes = plt.subplots(6, 1)
    f.suptitle("test_sgd_lr.01_max95acc")

    axes[0].set_title("layer0 - batch_act_diff between iterations")
    axes[1].set_title("layer1")
    axes[2].set_title("layer0 - batch_act_diff with all until of training")
    axes[3].set_title("layer1")
    axes[4].set_title("train_loss")
    axes[5].set_title("test_acc")
    axes[0].set_ylim([0, 1])
    axes[1].set_ylim([0, 1])
    axes[2].set_ylim([0, 0.1])
    axes[3].set_ylim([0, 0.1])

    axes[2].plot(common0e)
    axes[3].plot(common1e)
    axes[4].plot(loss)
    axes[5].plot(acc_idx, acc)

    for i in range(0, max_r, 10):
        axes[0].plot(common0a[i][1], common0a[i][0])
        axes[1].plot(common1a[i][1], common1a[i][0])
        plt.show()
        plt.waitforbuttonpress()
    plt.tight_layout()
    plt.show()
    plt.plot(common0)
    plt.plot(common1)
    plt.show()

    # Test model


def test(args, model, device, test_loader, masks=[None, None]):
    global test_stats
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, masks=masks)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_stats.append({
        "loss": test_loss,
        "accuracy": 100. * correct / len(test_loader.dataset),
    })
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def analyze_activations():
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    import pandas as pd

    data, target = next(iter(test_loader))
    output, s = models[i](data, stats=True, masks=masks)

    _, _, x0, x1 = s

    for ix in range(500):
        print(ix)
        plt.hist(x0[:, ix].data.numpy())
        plt.show()
        plt.waitforbuttonpress()
        plt.close()


def main():
    # Training settings
    from argparse import Namespace
    import pandas as pd
    import matplotlib.pyplot as plt

    args = Namespace
    args.batch_size = 128
    args.test_batch_size = 10000
    args.epochs = 10
    args.lr = 0.01
    args.momentum = 0.5
    args.no_cuda = False
    args.seed = 1
    args.log_interval = 10
    args.save_model = False

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    # optimizer = optim.Adam(model.parameters())
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.RMSprop(model.parameters())
    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.)

    if save_models:
        print("Saved model")
        new_model = Net().to(device)
        new_model.load_state_dict(model.state_dict())
        models[0] = new_model

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    torch.save({
        "saved_batch": saved_batch,
        "all_stats": all_stats,
        "test_stats": test_stats,
    }, "test_sgd_lr.01_max95acc")

    # ===========================================
    # test model with masks
    common0aV = dict()
    common1aV = dict()
    common0eV = dict()
    common1eV = dict()

    # ===========================================
    # get masks =
    data, target = next(iter(test_loader))
    data = data.cuda()
    target = target.cuda()
    output, s = models[midx](data, stats=True, masks=[None, None])
    _, _, _, _, x0, x1 = s
    x0m = x0.data.mean(dim=0).cuda()
    x1m = x1.data.mean(dim=0).cuda()


    # ===========================================

    # Models saved [0, 46, 93, 140, 187, 234, 281, 328, 375, 422, 469]
    data, target = next(iter(test_loader))
    data = data.cuda()
    target = target.cuda()

    midx = 187
    for midx in [0, 46, 93, 140, 187, 234, 281, 328, 375, 422]:
        # print("-"*10, midx)
        output, s = models[midx](data, stats=True, masks=[None, None])
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct_o = pred.eq(target.view_as(pred)).sum().item() / float(len(test_loader.dataset))
        _, _, _, _, x0, x1 = s
        x0m = x0.data.mean(dim=0).cuda()
        x1m = x1.data.mean(dim=0).cuda()

        masks = [((~common0eV[0]).cuda().float(), x0m), ((~common1eV[0]).cuda().float(), x1m)]
        output = models[midx](data, masks=masks)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item() / float(len(test_loader.dataset))
        # print("Correct", correct)
        print(f"[optim step {midx}] Acc: {correct} vs without mask {correct_o} "
              f"(no of act unchanged until end: "
              f"[{common0eV[midx].sum().item(), common1eV[midx].sum().item()}]")

        d = 1
        masks = [((~common0aV[midx][midx+d]).cuda().float(), x0m),
                 ((~common1aV[midx][midx+d]).cuda().float(), x1m)]
        output, s = models[midx](data, stats=True, masks=masks)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item() / float(len(test_loader.dataset))
        print(f"[optim step {midx}] Acc: {correct} vs without mask {correct_o} "
              f"(no of act unchanged until optim step +{d}: "
              f"[{common0aV[midx][midx+d].sum().item(), common1aV[midx][midx+d].sum().item()}]")

    test(args, models[midx], device, test_loader,
         masks=[((~common0eV[midx]).cuda().float(), x0m), ((~common0eV[midx]).cuda().float(), x1m)])

    m = torch.ones(500).cuda()
    for mi in range(500):
        m.fill_(1.)
        m[mi] = 0.
        print(f"Mask {mi}")
        test(args, models[midx], device, test_loader, masks=[m, None])

    # ==============================================================================================
    # Test no of important activations
    data, target = next(iter(test_loader))
    data = data.cuda()
    target = target.cuda()

    model_remove = {}
    models_keys = [46, 93, 140, 187, 234, 281, 328, 375, 422]
    with torch.no_grad():

        for midx in [46, 93, 140, 187, 234, 281, 328, 375, 422]:
            print(f"Model: {midx}")
            remove = {}
            models[midx].eval()
            output, s = models[midx](data, stats=True, masks=[None, None])
            _, _, _, _, x0, x1 = s
            x0m = x0.data.mean(dim=0).cuda()
            x1m = x1.data.mean(dim=0).cuda()

            for i in range(0, 500, 10):
                print(i)
                corrects = []
                for _ in range(50):
                    m1i = torch.randperm(500)[:i]
                    m1 = torch.ones(500)
                    m1[m1i] = 0.

                    m2i = torch.randperm(500)[:i]
                    m2 = torch.ones(500)
                    m2[m2i] = 0.

                    masks = [(m1.cuda().float(), x0m), (m2.cuda().float(), x1m)]

                    output = models[midx](data, masks=masks)

                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct = pred.eq(target.view_as(pred)).sum().item() / float(len(test_loader.dataset))
                    corrects.append(correct)
                remove[i] = corrects
            model_remove[midx] = remove

    dfs = {k: pd.DataFrame.from_dict(v) for k, v in model_remove.items()}

    step = 46
    ax = dfs[step].plot(kind="box", rot=90, title=f"RMSProp_default - optim step {step}")
    ax.set_ylim([0, 1])
    plt.xlabel('no of activations / layer - disabled (replaced with mean act)')
    plt.ylabel('test accuracy factor')
    plt.show()

    f, axes = plt.subplots(len(dfs),1 )
    f.suptitle("test_sgd_lr.01_max95acc")

    pidx = 0
    vs = []
    for k, v in dfs.items():
        v.plot(kind="box", ax=axes[pidx], rot=90)
        pidx += 1
        nv = v.copy()
        nv["model"] = k
        vs.append(nv)
    plt.show()

    df = pd.concat(vs)
    df.groupby("model").mean().transpose().plot()
    plt.show()

    df = pd.DataFrame(model_remove)
            # q = torch.zeros(500).cuda()
            # test(args, models[i], device, test_loader,
            #      masks=[(m1.cuda().float(), q), (m2.cuda().float(), q)])
            # test(args, models[i], device, test_loader,
            #      masks=[m1.cuda().float(), m2.cuda().float()])

    # ==============================================================================================

    # ===========================================
    for k, v in models.items():
        new_model = Net().to(device)
        new_model.load_state_dict(v.state_dict())
        models[k] = new_model

    # ===========================================

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == '__main__':
    main()
