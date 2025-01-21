from copy import copy
from dataclasses import dataclass
from typing import Sequence, Union

import jax
import jax.numpy as jnp

from rigno.utils import Array, ScalarArray
from rigno.dataset import Metadata

EPSILON = 1e-10

@dataclass
class BatchMetrics:
    mse: Array = None
    l1: Array = None
    l2: Array = None
    # Poseidon's l1-error
    _l1: float = None
    # Poseidon's l2-error
    _l2: float = None

    def map(self, f):
        for key in self.__dict__.keys():
            self.__setattr__(key, f(self.__getattribute__(key)))

    def reshape(self, shape):
        self.map(lambda m: m.reshape(shape))

    def __add__(self, obj):
      out = copy(self)
      for key in self.__dict__.keys():
        out.__setattr__(key, self.__getattribute__(key) + obj.__getattribute__(key))
      return out

@dataclass
class Metrics:
    mse: float = None
    l1: float = None
    l2: float = None
    # Poseidon's l1-error
    _l1: float = None
    # Poseidon's l2-error
    _l2: float = None

@dataclass
class EvalMetrics:
  direct_tau_frac: Metrics = None
  direct_tau_min: Metrics = None
  direct_tau_max: Metrics = None
  rollout: Metrics = None
  final: Metrics = None

  def to_dict(self):
      return {key: val.__dict__ for key, val in self.__dict__.items()}

def lp_norm(arr: Array, p: int = 2, chunks: Union[None, Sequence[int]] = None, num_chunks: int = None) -> Array:
    """
    Returns the Bochner Lp-norm of an array.

    Args:
        arr: Point-wise values on a uniform grid with the dimensions
            [batch, time, space, var]
        p: Order of the norm. Defaults to 2.
        chunks: Index of variable chunks for vectorial functions.
            If None, the entries of the last axis are interpreted as values of
            independent scalar-valued functions. Defaults to None.

    Returns:
        A scalar value for each sample in the batch [batch, *remaining_axes]
    """

    # Set the default chunks
    if chunks is None:
        chunks = jnp.arange(arr.shape[-1])
        num_chunks = arr.shape[-1]
        keep_var_dim = False
    else:
        keep_var_dim = True

    # Compute power of absolute value
    pow_abs = jnp.power(jnp.abs(arr), p)
    # Sum on timespace (quadrature)
    abs_pow_sum_vars = jnp.sum(pow_abs, axis=(1, 2))
    # Sum on variable chunks
    abs_pow_sum = jax.vmap(jax.ops.segment_sum, in_axes=(0, None, None))(abs_pow_sum_vars, chunks, num_chunks)
    # Take the p-th root
    pth_root = jnp.power(abs_pow_sum, (1/p))
    # Squeeze variable axis
    if not keep_var_dim:
        pth_root = jnp.squeeze(pth_root, axis=-1)

    return pth_root

def rel_lp_error(gtr: Array, prd: Array, p: int = 2, chunks: Union[None, Sequence[int]] = None, num_chunks: int = None) -> Array:
    """
    Returns the relative Bochner Lp-norm of an array with respect to a ground-truth.

    Args:
        gtr: Point-wise values of a ground-truth function on a uniform
            grid with the dimensions [batch, time, space, var]
        prd: Point-wise values of a predicted function on a uniform
            grid with the dimensions [batch, time, space, var]
        p: Order of the norm. Defaults to 2.
        chunks: Index of variable chunks for vectorial functions.
            If None, the entries of the last axis are interpreted as values of
            independent scalar-valued functions. Defaults to None.


    Returns:
        A scalar value for each sample in the batch [batch, var]
    """

    if chunks is None:
        chunks = jnp.arange(gtr.shape[-1])
        num_chunks = gtr.shape[-1]
    else:
        chunks = jnp.array(chunks)

    err = (prd - gtr)
    err_norm = lp_norm(err, p=p, chunks=chunks, num_chunks=num_chunks)
    gtr_norm = lp_norm(gtr, p=p, chunks=chunks, num_chunks=num_chunks)

    return (err_norm / (gtr_norm + EPSILON))

def rel_lp_error_norm(gtr: Array, prd: Array, p: int = 2, chunks: Union[None, Sequence[int]] = None, num_chunks: int = None) -> Array:
    """
    Returns the norm of the relative Bochner Lp-norm of an array with respect to a ground-truth.
    The entries of the last axis are interpreted as values of independent scalar-valued
    functions. This results in an error vector. The vector norm of the error vector is returned.

    Args:
        gtr: Point-wise values of a ground-truth function on a uniform
            grid with the dimensions [batch, time, space, var]
        prd: Point-wise values of a predicted function on a uniform
            grid with the dimensions [batch, time, space, var]
        p: Order of the norm. Defaults to 2.
        chunks: Index of variable chunks for vectorial functions.
            If None, the entries of the last axis are interpreted as values of
            independent scalar-valued functions. Defaults to None.

    Returns:
        The vector norm of the error vector [batch,]
    """

    err_per_var = rel_lp_error(gtr, prd, p=p, chunks=chunks, num_chunks=num_chunks)
    err_agg = jnp.linalg.norm(err_per_var, ord=p, axis=1)
    return err_agg

def rel_lp_error_mean(gtr: Array, prd: Array, p: int = 2, chunks: Union[None, Sequence[int]] = None, num_chunks: int = None) -> Array:
    """
    Returns the average of the relative Bochner Lp-norm of an array with respect to a ground-truth.
    The entries of the last axis are interpreted as values of independent scalar-valued
    functions. This results in an error vector. The vector norm of the error vector is returned.

    Args:
        gtr: Point-wise values of a ground-truth function on a uniform
            grid with the dimensions [batch, time, space, var]
        prd: Point-wise values of a predicted function on a uniform
            grid with the dimensions [batch, time, space, var]
        p: Order of the norm. Defaults to 2.
        chunks: Index of variable chunks for vectorial functions.
            If None, the entries of the last axis are interpreted as values of
            independent scalar-valued functions. Defaults to None.

    Returns:
        The vector norm of the error vector [batch,]
    """

    err_per_var = rel_lp_error(gtr, prd, p=p, chunks=chunks, num_chunks=num_chunks)
    err_agg = jnp.mean(err_per_var, axis=1)
    return err_agg

def rel_lp_loss(gtr: Array, prd: Array, p: int = 2) -> Array:
    """
    Returns the mean relative Bochner Lp-norm of an array with respect to a ground-truth.

    Args:
        gtr: Point-wise values of a ground-truth function on a uniform
            grid with the dimensions [batch, time, space, var]
        prd: Point-wise values of a predicted function on a uniform
            grid with the dimensions [batch, time, space, var]
        p: Order of the norm. Defaults to 2.

    Returns:
        Mean relative Lp-norm over the batch.
    """

    return jnp.mean(rel_lp_error_norm(gtr, prd, p=p))

def mse_error(gtr: Array, prd: Array) -> Array:
    """
    Returns the mean squared error per variable.
    All input shapes are [batch, time, space, var]
    Output shape is [batch,].
    """

    return jnp.mean(jnp.power(prd - gtr, 2), axis=(1, 2, 3))

def mse_loss(gtr: Array, prd: Array) -> ScalarArray:
    """
    Returns the mean squared error.
    All input shapes are [batch, time, space, var]
    Output shape is a scalar.
    """

    return jnp.mean(jnp.power(prd - gtr, 2))

def normalized_rel_lp_error_mean(gtr: Array, prd: Array, metadata: Metadata, p: int = 2) -> ScalarArray:
    """Compute mean relative lp-error of normalized vectors"""

    mean = jnp.array(metadata.global_mean)
    std = jnp.array(metadata.global_std)
    gtr_nrm = (gtr - mean) / std
    prd_nrm = (prd - mean) / std

    err = rel_lp_error_mean(
        gtr_nrm, prd_nrm, p=p,
        chunks=metadata.chunked_variables,
        num_chunks=metadata.num_variable_chunks,
    )

    return err
