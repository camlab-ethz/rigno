from typing import Union, NamedTuple

from flax import linen as nn

from rigno.utils import Array


class Inputs(NamedTuple):
  u: Array
  c: Union[Array, None]
  x_inp: Array
  x_out: Array
  t: Union[Array, float, None]
  tau: Union[Array, float, None]

class AbstractOperator(nn.Module):
  def setup(self):
    raise NotImplementedError

  def __call__(self,
    inputs: Inputs,
    **kwargs,
  ) -> Array:
    return self.call(inputs, **kwargs)

  def call(self, inputs: Inputs) -> Array:
    raise NotImplementedError

  @property
  def configs(self):
    configs = {
      attr: self.__getattr__(attr)
      for attr in self.__annotations__.keys() if attr != 'parent'
    }
    return configs
