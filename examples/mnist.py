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

class PoincarePlaneDistance(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            num_planes: int,  # out_features
            c=1.0,
            euclidean_inputs=True,
            rescale_euclidean_norms_gain=None,  # rescale euclidean norms based on the dimensions per space
            signed=True,
            scaled=True,
            squared=False,
            project_input=True,
            normal_std=None,
            dimensions_per_space=None,
            rescale_normal_params=False,
            effective_softmax_rescale=None,
            hyperbolic_representation_metric=None,
    ):
        super().__init__()
        self.euclidean_inputs = euclidean_inputs
        self.rescale_norms_gain = rescale_euclidean_norms_gain
        self.signed = signed
        self.scaled = scaled
        self.squared = squared
        self.project_input = project_input
        self.ball = geoopt.PoincareBall(c=c)
        self.in_features = in_features
        self.num_planes = num_planes
        self.rescale_normal_params = rescale_normal_params

        if effective_softmax_rescale is not None:
            if self.rescale_normal_params:
                self.logits_multiplier = effective_softmax_rescale
            else:
                self.logits_multiplier = effective_softmax_rescale * 2
        else:
            self.logits_multiplier = 1

        if dimensions_per_space is not None:
            assert in_features % dimensions_per_space == 0
            self.dimensions_per_space = dimensions_per_space
            self.num_spaces = in_features // dimensions_per_space
        else:
            self.dimensions_per_space = self.in_features
            self.num_spaces = 1

        self.normals = nn.Parameter(torch.empty((num_planes, self.num_spaces, self.dimensions_per_space)))
        self.bias = geoopt.ManifoldParameter(torch.zeros(num_planes, self.num_spaces, self.dimensions_per_space),
                                             manifold=self.ball)

        self.normal_std = normal_std
        self.reset_parameters()

        self.hyperbolic_representation_metric = hyperbolic_representation_metric
        if self.hyperbolic_representation_metric is not None and self.euclidean_inputs:
            self.hyperbolic_representation_metric.add('hyperbolic_representations')


    def get_mean_norm(self, input):
        if self.dimensions_per_space:
            input_shape = input.size()
            input_batch_dims = input_shape[:-1]
            input_feature_dim = input_shape[-1]
            rs_input = input.view(*input_batch_dims, input_feature_dim // self.dimensions_per_space,
                                  self.dimensions_per_space)
        else:
            rs_input = input
        return torch.norm(rs_input, p=2, dim=-1, keepdim=True).mean()

    def map_to_ball(self, input):  # input bs x in_feat
        if self.rescale_norms_gain:  # make expected tangent vector norm independent of initial dimension (approximately)
            input = self.rescale_norms_gain * input / np.sqrt(self.dimensions_per_space)
        return self.ball.expmap0(input, project=self.project_input)

    def manual_distance(self, points, other_points):
        dist = torch.arccosh(1 + 2 * (points - other_points).pow(2).sum(-1) / (1 - points.pow(2).sum(-1)) / (
                    1 - other_points.pow(2).sum(-1)))
        return dist

    def distance_matrix(self, input, euclidean_inputs=True, cpu=False):
        if euclidean_inputs:
            input = self.map_to_ball(input)
        input_batch_dims = input.size()[:-1]
        input = input.view(*input_batch_dims, self.num_spaces, self.dimensions_per_space)
        if cpu:
            input = input.cpu()
        distances = self.manual_distance(input.unsqueeze(0), input.unsqueeze(1))

        return distances.sum(-1)

    def distance_to_space(self, input, other, euclidean_inputs):
        if euclidean_inputs:
            input = self.map_to_ball(input)
            other = self.map_to_ball(other)
        input_batch_dims = input.size()[:-1]
        input = input.view(-1, self.num_spaces, self.dimensions_per_space)
        other = other.view(-1, self.num_spaces, self.dimensions_per_space)
        summed_dists = self.ball.dist(x=input, y=other).sum(-1)
        return summed_dists.view(input_batch_dims)

    def forward(self, input):  # input bs x in_feat
        input_batch_dims = input.size()[:-1]
        input = input.view(-1, self.num_spaces, self.dimensions_per_space)
        if self.euclidean_inputs:
            input = self.map_to_ball(input)
            if self.hyperbolic_representation_metric is not None:
                self.hyperbolic_representation_metric.set(hyperbolic_representations=input)
        input_p = input.unsqueeze(-3)  # bs x 1 x num_spaces x dim_per_space
        if self.rescale_normal_params:
            conformal_factor = 1 - self.bias.pow(2).sum(dim=-1)
            a = self.normals * conformal_factor.unsqueeze(-1)
        else:
            a = self.normals
        distances = self.ball.dist2plane(x=input_p, p=self.bias, a=a,
                                         signed=self.signed, scaled=self.scaled, dim=-1)
        if self.rescale_normal_params:
            distances = distances * 2 / conformal_factor
        distance = distances.sum(-1)
        distance = distance.view(*input_batch_dims, self.num_planes)
        return distance * self.logits_multiplier

    def forward_rs(self, input):  # input bs x in_feat
        input_batch_dims = input.size()[:-1]
        input = input.view(-1, self.num_spaces, self.dimensions_per_space)
        if self.euclidean_inputs:
            input = self.map_to_ball(input)
            if self.hyperbolic_representation_metric is not None:
                self.hyperbolic_representation_metric.set(hyperbolic_representations=input)
        input_p = input.unsqueeze(-3)  # bs x 1 x num_spaces x dim_per_space
        conformal_factor = 1 - self.bias.pow(2).sum(dim=-1)
        distances = self.ball.dist2plane(x=input_p, p=self.bias, a=self.normals * conformal_factor.unsqueeze(-1),
                                         signed=self.signed, scaled=self.scaled, dim=-1)
        distances = distances * 2 / conformal_factor
        distance = distances.sum(-1)
        distance = distance.view(*input_batch_dims, self.num_planes)
        return distance

    def extra_repr(self):
        return (
            "poincare_dim={num_spaces}x{dimensions_per_space} ({in_features}), "
            "num_planes={num_planes}, "
            .format(**self.__dict__))

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.zeros_(self.bias)
        if self.normal_std:
            nn.init.normal_(self.normals, std=self.normal_std)
        else:
            nn.init.normal_(self.normals, std=1 / np.sqrt(self.in_features))

def final_weight_init_hyp_small(m):
    if isinstance(m, PoincarePlaneDistance):
        nn.init.normal_(m.normals.data, 1 / np.sqrt(m.in_features) * 0.01)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

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
            # self.mlr = hypnn.HyperbolicMLR(ball_dim=args.dim, n_classes=10, c=args.c)
            self.mlr = PoincarePlaneDistance(in_features=args.dim, num_planes=10)
            # final_weight_init_hyp_small(self.mlr)

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
        hyp_x = None
        origin_dist = None
        if self.linear:
            return F.log_softmax(self.mlr(x), dim=-1), origin_dist, hyp_x
        else:
            # x = hyp_x
            # return F.log_softmax(self.mlr(x, c=self.tp.c), dim=-1), origin_dist, hyp_x
            return F.log_softmax(self.mlr(x), dim=-1), origin_dist, hyp_x


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
            # for label in test_loader.dataset.targets.unique():
            #     label_stats = origin_dist[target==int(label)].detach().cpu().numpy()
            #     label_ls = latents[target==int(label)].detach().cpu().numpy()
            #     label_dists = np.concatenate((origin_dist_stats[int(label)][0], label_stats))
            #     label_latents = np.concatenate((origin_dist_stats[int(label)][1], label_ls))
            #     origin_dist_stats[int(label)] = (label_dists, label_latents)

    test_loss /= len(test_loader.dataset)

    # for label in test_loader.dataset.targets.unique():
    #     print(int(label), np.mean(origin_dist_stats[int(label)][0]))

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
        "--dim", type=int, default=2, help="Dimension of the Poincare ball"
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
    
    # train_labels = [1, 2, 3, 4]
    train_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

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
    if args.linear:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=args.lr)

    test(args, model, device, test_loader, epoch=0, train_labels=train_labels)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, epoch=epoch, train_labels=train_labels)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
