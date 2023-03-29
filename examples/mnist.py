from __future__ import print_function

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

import hyptorch.nn as hypnn
import hyptorch.pmath as pmath
import wandb
import geoopt
from hyper_utils import PoincarePlaneDistance, ClipNorm, apply_sn

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, args.dim)
        # self.tp = hypnn.ToPoincare(
        #     c=args.c, train_x=args.train_x, train_c=args.train_c, ball_dim=args.dim
        # )
        self.linear = args.linear
        if self.linear:
            self.mlr = nn.Linear(args.dim, 10)
        else:
            # self.clip_norm = ClipNorm(args.clip_max_norm)
            self.poincare_affine = PoincarePlaneDistance(
                                                 euclidean_inputs=False,
                                                 in_features=args.dim, 
                                                 num_planes=10, 
                                                 c=args.c,
                                                 rescale_euclidean_norms_gain=args.rescale_euclidean_norms_gain, 
                                                 rescale_normal_params=False,
                                                 effective_softmax_rescale=args.effective_softmax_rescale
                                                 )
            # self.mlr = nn.Sequential(self.clip_norm, self.poincare_affine)
            self.mlr = nn.Sequential(self.poincare_affine)
        if not self.linear and not args.no_spectral_norm:
            for m in self.modules():
                apply_sn(m)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # hyp_x = self.tp(x)
        # origin_dist = pmath.dist0(hyp_x, c=self.tp.c)
        hyp_x = self.poincare_affine.map_to_ball(x)
        origin_dist = self.poincare_affine.ball.dist0(hyp_x)
        if self.linear:
            return F.log_softmax(self.mlr(x), dim=-1), origin_dist, hyp_x
        else:
            # return F.log_softmax(self.mlr(x, c=self.tp.c), dim=-1), origin_dist, hyp_x
            return F.log_softmax(self.mlr(hyp_x), dim=-1), origin_dist, hyp_x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, origin_dist, latents = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

def test(args, model, device, test_loader, epoch, train_labels):
    model.eval()
    test_loss = 0
    correct = 0
    origin_dist_stats = {}
    for label in test_loader.dataset.targets.unique():
        origin_dist_stats[int(label)] = (np.array([]), np.empty((1, args.dim)))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, origin_dist, latents = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            for label in test_loader.dataset.targets.unique():
                label_stats = origin_dist[target==int(label)].detach().cpu().numpy()
                label_ls = latents[target==int(label)].detach().cpu().numpy()
                label_dists = np.concatenate((origin_dist_stats[int(label)][0], label_stats))
                label_latents = np.concatenate((origin_dist_stats[int(label)][1], label_ls))
                origin_dist_stats[int(label)] = (label_dists, label_latents)

    test_loss /= len(test_loader.dataset)

    print("\nAverage Distance to Origin Per Class\n")
    for label in test_loader.dataset.targets.unique():
        print(int(label), f'{np.mean(origin_dist_stats[int(label)][0]):.2f}')

    # fig, ax = plt.subplots()
    # rng = np.random.default_rng(seed=0)
    # for i in range(10):
    #     points = origin_dist_stats[i][1][:50]
    #     # points = points[rng.integers(points.shape[0], 50, replace=False), :]
    #     label = i if i in train_labels else -1
    #     if label == -1:
    #         ax.scatter(points[:,0], points[:,1], label=label, alpha=0.3, c='black')
    #     else:
    #         ax.scatter(points[:,0], points[:,1], label=label, alpha=0.3)
    # plt.xlim(-1.2, 1.2)
    # plt.ylim(-1.2, 1.2)
    # fig.legend()
    # name_prefix = ''.join(str(y) for y in train_labels)
    # alg_type = 'linear' if args.linear else 'hyper'
    # fig.savefig(f'tmp{name_prefix}_{alg_type}_latents_plot_{epoch}.png')

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--linear",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--c", type=float, default=1.0, help="Curvature of the Poincare ball"
    )
    parser.add_argument(
        "--dim", type=int, default=4, help="Dimension of the Poincare ball"
    )
    parser.add_argument(
        "--train_x",
        action="store_true",
        default=False,
        help="train the exponential map origin",
    )
    parser.add_argument(
        "--train_c",
        action="store_true",
        default=False,
        help="train the Poincare ball curvature",
    )

    parser.add_argument(
        "--clip_max_norm", type=int, default=16
    )

    parser.add_argument(
        "--effective_softmax_rescale", type=int, default=2
    )

    parser.add_argument(
        "--rescale_euclidean_norms_gain", type=int, default=1
    )

    parser.add_argument(
        "--no_spectral_norm",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    if args.wandb:
        wandb.init(config=args, entity='mila-projects', project='hyper-mnist')

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_dataset = datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
    
    train_labels = [1, 2, 3, 7, 8, 9]
    # train_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    train_indices = torch.isin(train_dataset.targets, torch.Tensor(train_labels)).nonzero().squeeze()
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=False,
        **kwargs
    )

    model = Net(args).to(device)
    print(model)
    if args.linear:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        # optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=args.lr)
        optimizer = geoopt.optim.RiemannianSGD(model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)

    print(f"Training on classes {train_labels}, evaluating on all classes.")

    test(args, model, device, test_loader, epoch=0, train_labels=train_labels)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, epoch=epoch, train_labels=train_labels)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
