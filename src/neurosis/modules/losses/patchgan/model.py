from torch import Tensor, nn

from neurosis.modules.layers import ActNorm


def weights_init(m: nn.Module):
    init_gain = 0.02  # could make this a parameter, but it's not really necessary

    if isinstance(m, (nn.modules.conv._ConvNd, nn.Linear)) and hasattr(m, "weight"):
        nn.init.normal_(m.weight.data, 0.0, init_gain)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

    elif isinstance(m, nn.modules.batchnorm._BatchNorm):
        # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        nn.init.normal_(m.weight.data, 1.0, init_gain)
        nn.init.constant_(m.bias.data, 0.0)
    return


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        use_actnorm: bool = False,
    ):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            use_actnorm     -- whether to use ActNorm layers instead of BatchNorm
        """
        super().__init__()
        if use_actnorm:
            norm_layer = ActNorm
            use_bias = True
        else:
            norm_layer = nn.BatchNorm2d
            use_bias = False

        kernel_size = 4
        padding = 1
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(input_nc, ndf, kernel_size=kernel_size, stride=2, padding=padding),
                nn.LeakyReLU(0.2, True),
            ]
        )

        layer_mult = 1
        prev_layer_mult = 1
        for n in range(n_layers):  # gradually increase the number of filters
            layer_num = n + 1  # range() starts at 0, but we want to start at 1

            prev_layer_mult = layer_mult
            layer_mult = min(2**layer_num, 8)

            in_channels = ndf * prev_layer_mult
            out_channels = ndf * layer_mult

            self.layers.extend(
                [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=2 if layer_num < n_layers else 1,  # last layer has stride 1
                        padding=padding,
                        bias=use_bias,
                    ),
                    norm_layer(ndf * layer_mult),
                    nn.LeakyReLU(0.2, True),
                ]
            )

        # output 1 channel prediction map
        self.layers.append(
            nn.Conv2d(ndf * layer_mult, 1, kernel_size=kernel_size, stride=1, padding=padding),
        )

    def initialize_weights(self):
        return self.apply(weights_init)

    def forward(self, x: Tensor) -> Tensor:
        """Standard forward."""
        for layer in self.layers:
            x = layer(x)
        return x
