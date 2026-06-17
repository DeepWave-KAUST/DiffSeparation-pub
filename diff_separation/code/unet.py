from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    checkpoint,
)

class TimeEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for diffusion timesteps.

    In diffusion models, the denoising network must know the current noise level.
    The timestep t indicates how strongly the clean P-wave mode x_0 has been
    corrupted into x_t:

        x_t = sqrt(alpha_bar_t) x_0
              + sqrt(1 - alpha_bar_t) epsilon.

    This module maps the scalar timestep t into a high-dimensional sinusoidal
    embedding, which is then injected into each residual block of the U-Net.

    Parameters
    ----------
    dim : int
        Embedding dimension. It must be an even number because the embedding
        is built from sine and cosine pairs.

    scale : float
        Linear scaling factor applied to timesteps before constructing the
        sinusoidal embedding.

    Input
    -----
    x : torch.Tensor, shape [N]
        Batch of diffusion timesteps.

    Output
    ------
    torch.Tensor, shape [N, dim]
        Timestep embedding.
    """

    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2

        # Frequencies follow the standard transformer/DDPM sinusoidal embedding.
        emb = math.log(10000) / half_dim
        emb = th.exp(th.arange(half_dim, device=device) * -emb)

        # Outer product between timesteps and frequencies.
        emb = th.outer(x * self.scale, emb)

        # Concatenate sine and cosine components.
        emb = th.cat((emb.sin(), emb.cos()), dim=-1)

        return emb

class TimestepBlock(nn.Module):
    """
    Base class for modules conditioned on timestep embeddings.

    Any module inheriting from TimestepBlock expects a forward method of the form:

        forward(x, time_emb)

    This allows the U-Net to inject diffusion timestep information into residual
    blocks during denoising.
    """

    @abstractmethod
    def forward(self, x, time_emb):
        """
        Apply the module to x using timestep embedding time_emb.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    Sequential container that forwards timestep embeddings to supported layers.

    Standard layers such as convolution or attention only take x as input.
    Residual blocks require both x and the timestep embedding. This wrapper
    automatically checks whether each layer is a TimestepBlock and passes
    time_emb only when needed.
    """

    def forward(self, x, time_emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, time_emb)
            else:
                x = layer(x)

        return x


class Upsample(nn.Module):
    """
    Upsampling layer used in the decoder part of the U-Net.

    In the PICDM denoising network, the decoder progressively restores spatial
    resolution after the encoder compresses the feature maps. This helps recover
    fine-scale structures of the predicted clean P-wave mode.

    Parameters
    ----------
    channels : int
        Number of input and output feature channels.

    use_conv : bool
        If True, apply a convolution after nearest-neighbor upsampling.

    dims : int
        Signal dimensionality. The current wavefield problem uses dims=2.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims

        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels

        if self.dims == 3:
            x = F.interpolate(
                x,
                (x.shape[2], x.shape[3] * 2, x.shape[4] * 2),
                mode="nearest",
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")

        if self.use_conv:
            x = self.conv(x)

        return x


class Downsample(nn.Module):
    """
    Downsampling layer used in the encoder part of the U-Net.

    Downsampling increases the receptive field and allows the network to learn
    large-scale wavefield structures. This is important for elastic wave-mode
    separation because P/S wave patterns may have nonlocal spatial correlations.

    Parameters
    ----------
    channels : int
        Number of input and output feature channels.

    use_conv : bool
        If True, use a strided convolution. Otherwise, use average pooling.

    dims : int
        Signal dimensionality. The current wavefield problem uses dims=2.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims

        stride = 2 if dims != 3 else (1, 2, 2)

        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    Residual block with diffusion timestep conditioning.

    This is the basic feature-extraction unit of the U-Net. The block first
    processes the input feature map with normalization, SiLU activation, and
    convolution. It then injects the timestep embedding to inform the block
    about the current diffusion noise level.

    In PICDM, the same denoising network is used for all timesteps. Therefore,
    timestep conditioning is essential: it tells the network whether it is
    denoising a weakly corrupted P-wave mode or a nearly Gaussian-noise sample.

    Parameters
    ----------
    channels : int
        Number of input channels.

    time_emb_channels : int
        Dimension of the timestep embedding.

    dropout : float
        Dropout probability. In the paper's default configuration, dropout=0.0.

    out_channels : int, optional
        Number of output channels. If None, it equals channels.

    use_conv : bool
        If True and out_channels differs from channels, use a spatial
        convolution in the skip connection.

    use_scale_shift_norm : bool
        If True, use scale-shift normalization. This follows improved diffusion
        and allows timestep embeddings to modulate normalized features by
        predicted scale and shift.

    dims : int
        Spatial dimensionality.

    use_checkpoint : bool
        If True, use gradient checkpointing to reduce memory usage.
    """

    def __init__(
        self,
        channels,
        time_emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()

        self.channels = channels
        self.time_emb_channels = time_emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        # Main input transformation.
        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        # Project timestep embedding to feature-channel dimension.
        # If scale-shift normalization is used, the embedding predicts both
        # scale and shift, hence 2 * out_channels.
        self.time_emb_layers = nn.Sequential(
            SiLU(),
            linear(
                time_emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        # Output transformation. The final convolution is initialized to zero,
        # which stabilizes diffusion-model training at initialization.
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        # Skip connection. This preserves information and improves gradient flow.
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims,
                channels,
                self.out_channels,
                3,
                padding=1,
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, time_emb, cls_emb=None):
        """
        Apply the residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map.

        time_emb : torch.Tensor
            Diffusion timestep embedding.

        cls_emb : torch.Tensor, optional
            Unused placeholder for compatibility.

        Returns
        -------
        torch.Tensor
            Output feature map.
        """
        return checkpoint(
            self._forward,
            (x, time_emb),
            self.parameters(),
            self.use_checkpoint,
        )

    def _forward(self, x, time_emb):
        h = self.in_layers(x)

        # Convert timestep embedding to feature modulation parameters.
        time_emb_out = self.time_emb_layers(time_emb).type(h.dtype)

        # Broadcast timestep embedding over spatial dimensions.
        while len(time_emb_out.shape) < len(h.shape):
            time_emb_out = time_emb_out[..., None]

        if self.use_scale_shift_norm:
            # Scale-shift normalization:
            # normalized features are modulated as h_norm * (1 + scale) + shift.
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(time_emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            # Add timestep embedding directly to feature maps.
            h = h + time_emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    Spatial self-attention block.

    This block allows different spatial positions in the wavefield feature map
    to attend to each other. For elastic wave-mode separation, this helps the
    network capture long-range wavefield structures and coherent wavefronts that
    cannot be fully represented by local convolutions alone.

    Parameters
    ----------
    channels : int
        Number of feature channels.

    num_heads : int
        Number of attention heads.

    use_checkpoint : bool
        If True, use gradient checkpointing.
    """

    def __init__(self, channels, num_heads=4, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)

        # 1D convolution over flattened spatial positions to produce Q, K, V.
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()

        # Zero-initialized output projection for stable training.
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape

        # Flatten spatial dimensions: [B, C, H, W] -> [B, C, H*W].
        x = x.reshape(b, c, -1)

        qkv = self.qkv(self.norm(x))

        # Merge batch and attention heads for multi-head attention.
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])

        h = self.attention(qkv)

        # Restore batch dimension.
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)

        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    Multi-head QKV attention.

    The input contains concatenated query, key, and value tensors. The attention
    operation computes spatial interactions and returns an attention-weighted
    feature representation.
    """

    def forward(self, qkv):
        """
        Apply scaled dot-product attention.

        Parameters
        ----------
        qkv : torch.Tensor, shape [B * num_heads, 3C, T]
            Concatenated query, key, and value tensors, where T is the number
            of flattened spatial positions.

        Returns
        -------
        torch.Tensor, shape [B * num_heads, C, T]
            Attention output.
        """
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)

        # More stable for fp16 than dividing the attention score afterwards.
        scale = 1 / math.sqrt(math.sqrt(ch))

        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)

        return th.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        Count FLOPs for the QKV attention operation.

        This helper is intended for use with the thop package and is not needed
        during standard PICDM training or inference.
        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))

        # Two matrix multiplications are used:
        # 1. query-key attention scores,
        # 2. attention-weighted value aggregation.
        matmul_ops = 2 * b * (num_spatial ** 2) * c

        model.total_ops += th.DoubleTensor([matmul_ops])


class UNetModel(nn.Module):
    """
    Conditional U-Net denoising network for PICDM.

    This is the main neural network used in the physics-informed conditional
    diffusion model for elastic wave-mode separation.

    Network role in the paper
    -------------------------
    The diffusion model corrupts the clean P-wave mode x_0 into a noisy version
    x_t. The U-Net learns the reverse denoising map:

        f_theta(x_t, t, c) -> x_0,

    where:

        x_t = noisy P-wave mode,
        t   = diffusion timestep,
        c   = physical condition.

    In this implementation, the physical condition is provided by direct channel
    concatenation:

        c = (Vx, Vz, vp, vs).

    Therefore, the actual U-Net input is:

        concat(x_t, Vx, Vz, vp, vs),

    which has 6 channels. The output is the predicted clean P-wave mode:

        (Vx^p, Vz^p),

    which has 2 channels.

    Architecture
    ------------
    The default setting in script_util.py corresponds to the architecture
    described in the paper:

        - five-scale U-Net,
        - base channel width 64,
        - channel widths 64, 128, 256, 512, 1024,
        - two residual blocks at each scale,
        - timestep embedding injected into residual blocks,
        - self-attention at selected resolutions,
        - skip connections between encoder and decoder,
        - final 3 x 3 convolution to output two P-wave components.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        time_emb_scale=1.0,
        num_classes=None,
        use_checkpoint=False,
        num_heads=4,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        # Pad input sizes to be divisible by 2 ** len(channel_mult), so that
        # repeated downsampling and upsampling are valid even for arbitrary
        # wavefield dimensions.
        self.padder_size = 2 ** len(channel_mult)

        # Timestep embedding network. The scalar diffusion timestep is first
        # encoded by TimeEmbedding and then projected through an MLP.
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            TimeEmbedding(model_channels, time_emb_scale),
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # Initial convolution from input channels to base feature channels.
        self.inp = conv_nd(dims, in_channels, model_channels, 3, padding=1)

        # Encoder/downsampling path.
        self.downs = nn.ModuleList([])
        encoder_channels = [model_channels]

        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]

                ch = mult * model_channels

                # Insert attention at selected downsampling rates.
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                        )
                    )

                self.downs.append(TimestepEmbedSequential(*layers))
                encoder_channels.append(ch)

            # Downsample between scales except after the last scale.
            if level != len(channel_mult) - 1:
                self.downs.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                encoder_channels.append(ch)
                ds *= 2

        # Bottleneck/middle block.
        # It combines residual processing and attention at the lowest resolution.
        self.middle = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        # Decoder/upsampling path.
        self.ups = nn.ModuleList([])

        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                # Concatenate current decoder features with the corresponding
                # encoder skip features. Therefore, input channels are:
                #     current channels + skip channels.
                layers = [
                    ResBlock(
                        ch + encoder_channels.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]

                ch = model_channels * mult

                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                        )
                    )

                # Upsample at the end of each decoder scale except the final one.
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2

                self.ups.append(TimestepEmbedSequential(*layers))

        # Final output projection to the two-channel clean P-wave mode.
        # zero_module initializes the final convolution to zero, which is common
        # in diffusion U-Nets for stable initial behavior.
        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the main U-Net body to float16.

        This is used only when mixed-precision training is enabled. The input
        and output layers can remain in fp32 while the main torso is converted
        to reduce memory usage and improve speed.
        """
        self.downs.apply(convert_module_to_f16)
        self.middle.apply(convert_module_to_f16)
        self.ups.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the main U-Net body back to float32.
        """
        self.downs.apply(convert_module_to_f32)
        self.middle.apply(convert_module_to_f32)
        self.ups.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Return the dtype used by the main U-Net body.
        """
        return next(self.downs.parameters()).dtype

    def forward(self, x_t, vxz, vel, loc, timesteps, y=None):
        """
        Forward pass of the PICDM conditional U-Net.

        Parameters
        ----------
        x_t : torch.Tensor, shape [B, 2, H, W]
            Noisy P-wave mode at diffusion timestep t. Its two channels are:
                x_t[:, 0] = noisy Vx^p,
                x_t[:, 1] = noisy Vz^p.

        vxz : torch.Tensor, shape [B, 2, H, W]
            Original coupled elastic velocity wavefield:
                vxz[:, 0] = Vx,
                vxz[:, 1] = Vz.

            This is part of the physical condition c.

        vel : torch.Tensor, shape [B, 2, H, W]
            Velocity-model condition:
                vel[:, 0] = normalized vp,
                vel[:, 1] = normalized vs.

            The paper shows that velocity conditioning improves generalization,
            especially for out-of-distribution velocity structures.

        loc : torch.Tensor
            Source location and snapshot-time metadata. It is included in the
            function interface for compatibility and possible future conditioning,
            but it is not explicitly used in the current forward pass.

        timesteps : torch.Tensor, shape [B]
            Diffusion timestep indices. These are embedded and injected into
            all residual blocks.

        y : torch.Tensor, optional
            Class labels for class-conditional diffusion. Not used in PICDM.

        Returns
        -------
        torch.Tensor, shape [B, 2, H, W]
            Predicted clean P-wave mode:
                output[:, 0] = predicted Vx^p,
                output[:, 1] = predicted Vz^p.
        """
        b, c, h, w = x_t.shape

        # Concatenate noisy target and physical conditions along the channel axis.
        #
        # Channel layout:
        #   0-1: noisy P-wave mode x_t = (Vx^p, Vz^p)
        #   2-3: original elastic wavefield (Vx, Vz)
        #   4-5: velocity models (vp, vs)
        #
        # This gives the six-channel input used in the paper.
        x = th.cat([x_t, vxz, vel], dim=1)

        # Pad spatial dimensions if needed so that the U-Net downsampling path
        # can divide the image/wavefield size by powers of 2.
        x = self.check_image_size(x)

        # Embed diffusion timesteps.
        time_emb = self.time_embed(timesteps)

        skips = []

        # Convert feature tensor to the dtype used by the U-Net body.
        x = x.type(self.inner_dtype)

        # Initial convolution.
        x = self.inp(x)
        skips.append(x)

        # Encoder path.
        for module in self.downs:
            x = module(x, time_emb)
            skips.append(x)

        # Bottleneck.
        x = self.middle(x, time_emb)

        # Decoder path with skip connections.
        for module in self.ups:
            cat_in = th.cat([x, skips.pop()], dim=1)
            x = module(cat_in, time_emb)

        # Convert back to the dtype of the input x_t.
        x = x.type(x_t.dtype)

        # Final two-channel prediction of the clean P-wave mode.
        x = self.out(x)

        # Remove any padding and return the original spatial size.
        return x[:, :, :h, :w]

    def check_image_size(self, x):
        """
        Pad the input wavefield so that its spatial size is compatible with U-Net.

        Because the U-Net repeatedly downsamples the spatial dimensions, H and W
        should be divisible by 2 ** number_of_scales. If not, this function pads
        the right and bottom boundaries using replicate padding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape [B, C, H, W].

        Returns
        -------
        torch.Tensor
            Padded tensor.
        """
        _, _, h, w = x.size()

        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size

        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode="replicate")

        return x
