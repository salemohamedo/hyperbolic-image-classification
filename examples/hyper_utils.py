import geoopt
import torch.nn as nn
import torch
import numpy as np

from torch.nn.utils.parametrizations import spectral_norm

class ClipNorm(nn.Module):
    def __init__(self, max_norm=15, dimensions_per_space=None):
        super().__init__()
        self.max_norm = max_norm
        self.dimension_per_space = dimensions_per_space

    def get_mean_norm(self, input):
        if self.dimension_per_space:
            input_shape = input.size()
            input_batch_dims = input_shape[:-1]
            input_feature_dim = input_shape[-1]
            rs_input = input.view(*input_batch_dims, input_feature_dim // self.dimension_per_space,
                                  self.dimension_per_space)
        else:
            rs_input = input
        return torch.norm(rs_input, p=2, dim=-1, keepdim=True).mean()

    def forward(self, input):  # input bs x in_feat
        if self.dimension_per_space:
            input_shape = input.size()
            input_batch_dims = input_shape[:-1]
            input_feature_dim = input_shape[-1]
            rs_input = input.view(*input_batch_dims, input_feature_dim // self.dimension_per_space,
                                  self.dimension_per_space)
        else:
            rs_input = input
        input_l2 = torch.norm(rs_input, p=2, dim=-1, keepdim=True)
        clipped_input = torch.minimum(self.max_norm / input_l2,
                                      torch.ones_like(input_l2)) * rs_input
        if self.dimension_per_space:
            clipped_input = clipped_input.view(*input_shape)
        return clipped_input


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
            normal_std_scale=None,
            dimensions_per_space=None,
            rescale_normal_params=False,
            # Setting to true fixes (?) the logits magnitude over geoopt's default (which should be smaller by a factor of 2)
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
                # the implementation not rescaling should already be smaller by a factor of 2 due to geoopt issue
                self.logits_multiplier = effective_softmax_rescale * 2
        else:
            self.logits_multiplier = 1

        if dimensions_per_space is not None:
            assert in_features % dimensions_per_space == 0
            self.dimensions_per_space = dimensions_per_space
            self.num_spaces = in_features // dimensions_per_space
            # balls = [(self.ball, self.dimensions_per_space) for _ in range(self.num_spaces)]
            # self.manifold = geoopt.StereographicProductManifold(*balls)
        else:
            self.dimensions_per_space = self.in_features
            self.num_spaces = 1
            # self.manifold = self.ball

        # self.normals = nn.Parameter(torch.empty(num_planes, self.in_features)) # SLOW
        # self.bias = geoopt.ManifoldParameter(torch.zeros(num_planes, self.in_features), manifold=self.manifold)

        self.normals = nn.Parameter(torch.empty((num_planes, self.num_spaces, self.dimensions_per_space)))
        self.bias = geoopt.ManifoldParameter(torch.zeros(num_planes, self.num_spaces, self.dimensions_per_space),
                                             manifold=self.ball)

        self.normal_std_scale = normal_std_scale
        self.reset_parameters()

        self.hyperbolic_representation_metric = hyperbolic_representation_metric
        if self.hyperbolic_representation_metric is not None and self.euclidean_inputs:
            self.hyperbolic_representation_metric.add('hyperbolic_representations')
            self.hyperbolic_representation_metric.add('pre_hyp_euclidean_representations')

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
        if self.hyperbolic_representation_metric is not None:
                self.hyperbolic_representation_metric.set(pre_hyp_euclidean_representations=input)
        return self.ball.expmap0(input, project=self.project_input)

    def manual_distance(self, points, other_points):
        dist = torch.arccosh(1 + 2 * (points - other_points).pow(2).sum(-1) / (1 - points.pow(2).sum(-1)) / (
                    1 - other_points.pow(2).sum(-1)))
        return dist

    def distance_matrix(self, input, euclidean_inputs=True, cpu=False):
        if euclidean_inputs:
            input = self.map_to_ball(input)
        # print(input[0])
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
        # distance = torch.norm(distances, p=2, dim=-1) # wrong, we are loosing sign!
        # if self.signed
        # We have a different norm per each element of the product space - dot product should just SUM the logits of the different dimensions
        if self.rescale_normal_params:
            distances = distances * 2 / conformal_factor
        distance = distances.sum(-1)
        distance = distance.view(*input_batch_dims, self.num_planes)
        return distance * self.logits_multiplier

    def extra_repr(self):
        return (
            "poincare_dim={num_spaces}x{dimensions_per_space} ({in_features}), "
            "num_planes={num_planes}, "
            .format(**self.__dict__))

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.zeros_(self.bias)
        if self.normal_std_scale:
            nn.init.normal_(self.normals, std=1 / np.sqrt(
                self.in_features)*self.normal_std_scale)
        else:
            nn.init.normal_(self.normals, std=1 / np.sqrt(self.in_features))

def apply_sn(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        return spectral_norm(m)
    else:
        return m
