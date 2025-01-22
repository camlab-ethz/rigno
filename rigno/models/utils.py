"""Utility functions that are used in."""

from typing import Sequence, Callable

import flax.linen as nn
import jax.numpy as jnp
import jax.tree_util as tree


def concatenate_args(args, kwargs, axis: int = -1):
  """Concatenates all positional and keyword arguments on the given axis."""

  combined_args = tree.tree_flatten(args)[0] + tree.tree_flatten(kwargs)[0]
  concat_args = jnp.concatenate(combined_args, axis=axis)
  return concat_args

class FeedForwardBlock(nn.Module):
  """
  Multi-layer perceptron with optional layer norm and learned distribution correction
  on the last layer. Activation is applied on all layers except the last one. Multiple
  inputs are concatenated before being fed to the MLP.
  """

  layer_sizes: Sequence[int]
  activation: Callable
  use_layer_norm: bool = False
  use_conditional_norm: bool = False
  cond_norm_hidden_size: int = 4
  concatenate_axis: int = -1

  def setup(self):
    # Set up layers
    self.layers = [nn.Dense(features) for features in self.layer_sizes]

    # Set up normalization layer
    self.layernorm = nn.LayerNorm(
      reduction_axes=-1,
      feature_axes=-1,
      use_scale=True,
      use_bias=True,
    ) if self.use_layer_norm else None

    # Set conditional normalization layer
    self.correction = None
    if self.use_conditional_norm:
      self.correction = ConditionedNorm(
        latent_size=self.cond_norm_hidden_size,
        correction_size=self.layer_sizes[-1],
      )

  def __call__(self, *args, c = None, **kwargs):
    x = concatenate_args(args=args, kwargs=kwargs, axis=self.concatenate_axis)
    for layer in self.layers[:-1]:
      x = layer(x)
      x = self.activation(x)
    x = self.layers[-1](x)
    if self.layernorm:
      x = self.layernorm(x)
    if self.correction:
      assert c is not None
      x = self.correction(c=c, x=x)
    return x

class ConditionedNorm(nn.Module):
  """
  Learned correction layer is designed to be applied after a normalization layer.
  Based on an input (e.g., lead time), it shifts and scales the distribution of its input.
  correction_size must either be 1 or the same as one of the input dimensions (broadcastable).
  """

  latent_size: Sequence[int]
  correction_size: int = 1

  def setup(self):
    self.mlp_scale = nn.Sequential(
      layers=[
        nn.Dense(self.latent_size,
                 kernel_init=nn.initializers.normal(stddev=.01)),
        nn.sigmoid,
        nn.Dense(self.correction_size,
                 kernel_init=nn.initializers.normal(stddev=.01)),
      ])
    self.mlp_bias = nn.Sequential(
      layers=[
        nn.Dense(self.latent_size,
                 kernel_init=nn.initializers.normal(stddev=.01)),
        nn.sigmoid,
        nn.Dense(self.correction_size,
                 kernel_init=nn.initializers.normal(stddev=.01)),
      ])

  def __call__(self, c, x):
    scale = 1 + c * self.mlp_scale(c)
    bias = c * self.mlp_bias(c)
    shape = x.shape
    x = x.reshape(shape[0], -1, shape[-1])
    scale = jnp.expand_dims(scale, axis=1)
    bias = jnp.expand_dims(bias, axis=1)
    x = x * scale + bias
    x = x.reshape(*shape)
    return x
