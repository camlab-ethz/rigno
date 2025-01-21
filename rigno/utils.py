from absl import logging
from time import time
from typing import Union, Sequence, Callable

import flax.typing
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np


Array = Union[jnp.ndarray, np.ndarray]
ScalarArray = Union[jnp.ndarray, np.ndarray]

class disable_logging:
  """Context manager for disabling the logging."""

  def __init__(self, level: int = -1):
    self.level_context = level
    self.level_init = None

  def __enter__(self):
    self.level_init = logging.get_verbosity()
    logging.set_verbosity(self.level_context)

  def __exit__(self, exc_type, exc_value, traceback):
    logging.set_verbosity(self.level_init)

def is_multiple(b, a):
  return abs(int(b / a) * a - b) < 1e-08

def profile(f: Callable, kwargs: dict, repeats: int = 1, block_until_ready: bool = False):
  t_0 = time()
  for _ in range(repeats):
    u = f(**kwargs)
  if block_until_ready:
    u = u.block_until_ready()
  return ((time() - t_0) / repeats)

def shuffle_arrays(key: flax.typing.PRNGKey, arrays: Sequence[Array], axis: int = 0) -> Sequence[Array]:
  """Shuffles a set of arrays with the same random permutation along the leading axis."""

  # Move the desired axis to the leading axis
  arrays = tree.tree_map(lambda v: jnp.moveaxis(v, axis, 0), arrays)

  # Get permutation
  length = arrays[0].shape[0]
  assert all(tree.tree_map(lambda v: v.shape[0] == length, arrays))
  permutation = jax.random.permutation(key, length)

  # Permute along the leading axis
  arrays = tree.tree_map(lambda v: v[permutation], arrays)
  # Move back the leading axis to its place
  arrays = tree.tree_map(lambda v: jnp.moveaxis(v, 0, axis), arrays)

  return arrays

def split_arrays(arrays: Sequence[Array], size: int) -> Sequence[Array]:

  length = arrays[0].shape[0]
  assert all([arr.shape[0] == length for arr in arrays])

  return [jnp.stack(jnp.split(arr, length // size)) for arr in arrays]

def normalize(arr: Array, shift: Array, scale: Array):
  scale = jnp.where(scale == 0., 1., scale)
  arr = (arr - shift) / scale
  return arr

def unnormalize(arr: Array, mean: Array, std: Array):
  arr = std * arr + mean
  return arr
