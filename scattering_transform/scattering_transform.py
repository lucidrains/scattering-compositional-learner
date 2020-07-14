import torch
from torch import nn
import torch.nn.functional as F

# simple MLP with ReLU activation

class MLP(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        assert len(dims) >= 3, 'must have at least 3 dimensions, for dimension in and dimension out'

        layers = []
        pairs = list(zip(dims[:-1], dims[1:]))

        for ind, (dim_in, dim_out) in enumerate(pairs):
            is_last = ind >= (len(pairs) - 1)
            layers.append(nn.Linear(dim_in, dim_out))
            if not is_last:
                layers.append(nn.ReLU(inplace = True))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# the feedforward residual block mentioned in the paper
# used after extracting the visual features, as well as post-extraction of attribute information

class FeedForwardResidual(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.LayerNorm(dim * mult),
            nn.ReLU(inplace = True),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return x + self.net(x)

# convolutional net
# todo, make customizable and add Evonorm for batch independent normalization

class ConvNet(nn.Module):
    def __init__(self, image_size, chans, output_dim):
        super().__init__()

        num_conv_layers = len(chans) - 1
        conv_output_size = image_size // (2 ** num_conv_layers)

        convolutions = []
        channel_pairs = list(zip(chans[:-1], chans[1:]))

        for chan_in, chan_out in channel_pairs:
            convolutions.append(nn.Conv2d(chan_in, chan_out, 3, padding=1, stride=2))

        self.net = nn.Sequential(
            *convolutions,
            nn.Flatten(1),
            nn.Linear(chans[-1] * (conv_output_size ** 2), output_dim),
            nn.ReLU(inplace=True),
            FeedForwardResidual(output_dim)
        )

    def forward(self, x):
        return self.net(x)

# main scattering compositional learner class

class SCL(nn.Module):
    def __init__(
        self,
        image_size = 160,
        set_size = 9,
        conv_channels = [1, 16, 16, 32, 32, 32],
        conv_output_dim = 80,
        attr_heads = 10,
        attr_net_hidden_dims = [128],
        rel_heads = 80,
        rel_net_hidden_dims = [64, 23, 5]):

        super().__init__()
        self.vision = ConvNet(image_size, conv_channels, conv_output_dim)

        self.attr_heads = attr_heads
        attr_dim = conv_output_dim // attr_heads

        self.attr_net = MLP(attr_dim, *attr_net_hidden_dims, attr_dim)
        self.ff_residual = FeedForwardResidual(conv_output_dim)

        self.rel_heads = rel_heads
        self.rel_net = MLP(set_size, *rel_net_hidden_dims)

        self.to_logit = nn.Linear(rel_net_hidden_dims[-1] * rel_heads, 1)

    def forward(self, sets):
        b, c, n, h, w = sets.shape
        images = sets.view(-1, 1, h, w)
        features = self.vision(images)

        features = features.reshape(features.shape[0], self.attr_heads, -1)
        attrs = self.attr_net(features)
        attrs = attrs.reshape(b, c, n, -1)
        attrs = self.ff_residual(attrs)

        attrs = attrs.reshape(b, c, -1, self.rel_heads).transpose(-1, -2)
        rels = self.rel_net(attrs)
        rels = rels.flatten(2)
        
        logits = self.to_logit(rels).flatten(1)
        return logits
