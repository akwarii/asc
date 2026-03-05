import inspect
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)


class MLP(torch.nn.Module):
    r"""A Multi-Layer Perception (MLP) model.

    There exists two ways to instantiate an :class:`MLP`:

    1. By specifying explicit channel sizes, *e.g.*,

       .. code-block:: python

          mlp = MLP([16, 32, 64, 128])

       creates a three-layer MLP with **differently** sized hidden layers.

    1. By specifying fixed hidden channel sizes over a number of layers,
       *e.g.*,

       .. code-block:: python

          mlp = MLP(in_channels=16, hidden_channels=32, out_channels=128, num_layers=3)

       creates a three-layer MLP with **equally** sized hidden layers.

    Args:
        channel_list (list[int] or int, optional): list of input, intermediate
            and output channels such that :obj:`len(channel_list) - 1` denotes
            the number of layers of the MLP (default: :obj:`None`)
        in_channels (int, optional): Size of each input sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        hidden_channels (int, optional): Size of each hidden sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        out_channels (int, optional): Size of each output sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        num_layers (int, optional): The number of layers.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        dropout (float): Dropout probability of each hidden embedding.
            (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`"batch_norm"`)
        norm_kwargs (dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        plain_last (bool, optional): If set to :obj:`False`, will apply
            non-linearity, batch normalization and dropout to the last layer as
            well. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layers will not
            learn an additive bias. (default: :obj:`True`)
    """

    supports_norm_batch: bool

    def __init__(
        self,
        channel_list: list[int] | int | None = None,
        *,
        in_channels: int | None = None,
        hidden_channels: int | None = None,
        out_channels: int | None = None,
        num_layers: int | None = None,
        dropout: float = 0.0,
        act: str | Callable | None = "relu",
        act_first: bool = False,
        act_kwargs: dict[str, Any] | None = None,
        norm: str | Callable | None = "batch_norm",
        norm_kwargs: dict[str, Any] | None = None,
        plain_last: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if isinstance(channel_list, int):
            in_channels = channel_list

        if in_channels is not None:
            if num_layers is None:
                raise ValueError("Argument `num_layers` must be given")
            if num_layers > 1 and hidden_channels is None:
                raise ValueError(
                    f"Argument `hidden_channels` must be given for `num_layers={num_layers}`"
                )
            if out_channels is None:
                raise ValueError("Argument `out_channels` must be given")

            if num_layers == 1:
                channel_list = [in_channels, out_channels]
            else:
                assert hidden_channels is not None
                channel_list = [hidden_channels] * (num_layers - 1)
                channel_list = [in_channels] + channel_list + [out_channels]

        assert isinstance(channel_list, list)
        assert len(channel_list) >= 2
        self.channel_list = channel_list

        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.act_first = act_first
        self.plain_last = plain_last
        self.dropout = dropout

        self.linears = nn.ModuleList()
        for in_channels, out_channels in zip(channel_list[:-1], channel_list[1:]):
            self.linears.append(Linear(in_channels, out_channels, bias=bias))

        self.norms = nn.ModuleList()
        iterator = channel_list[1:-1] if plain_last else channel_list[1:]
        for hidden_channels in iterator:
            if norm is not None:
                norm_layer = normalization_resolver(
                    norm,
                    hidden_channels,
                    **(norm_kwargs or {}),
                )
            else:
                norm_layer = nn.Identity()
            self.norms.append(norm_layer)

        self.supports_norm_batch = False
        if len(self.norms) > 0 and hasattr(self.norms[0], "forward"):
            norm_params = inspect.signature(self.norms[0].forward).parameters
            self.supports_norm_batch = "batch" in norm_params

        self.reset_parameters()

    @property
    def in_channels(self) -> int:
        r"""Size of each input sample."""
        return self.channel_list[0]

    @property
    def out_channels(self) -> int:
        r"""Size of each output sample."""
        return self.channel_list[-1]

    @property
    def num_layers(self) -> int:
        r"""The number of layers."""
        return len(self.channel_list) - 1

    def reset_parameters(self) -> None:
        r"""Resets all learnable parameters of the module."""
        for linear in self.linears:
            linear.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()

    def forward(
        self,
        x: Tensor,
        batch: Tensor | None = None,
        batch_size: int | None = None,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            x (Tensor): The source tensor.
            batch (Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
        """
        # If `plain_last=True`, then `len(norms) = len(linears) -1, thus skipping
        # the execution of the last layer inside the for-loop.
        for linear, norm in zip(self.linears, self.norms):
            x = linear(x)
            if self.act is not None and self.act_first:
                x = self.act(x)
            if self.supports_norm_batch:
                x = norm(x, batch, batch_size)
            else:
                x = norm(x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final layer (if plain_last=True, this is skipped in the for-loop)
        if self.plain_last:
            x = self.linears[-1](x)

        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self.channel_list)[1:-1]})"
