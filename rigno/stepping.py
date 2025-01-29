"""Time-stepping classes."""

from abc import ABC, abstractmethod
from typing import Union

import flax.typing
import jax
import jax.numpy as jnp

from rigno.models.operator import AbstractOperator, Inputs
from rigno.utils import Array, is_multiple, normalize, unnormalize

class Stepper(ABC):
  """Abstract class for a time-stepping scheme."""

  def __init__(self, operator: AbstractOperator):
    self._apply_operator = operator.apply

  def normalize_inputs(self, stats, inputs: Inputs) -> Inputs:
    """Normalizes the inputs using the given statistics."""

    u_nrm = normalize(inputs.u, shift=stats['u']['mean'], scale=stats['u']['std'])
    if inputs.c is None:
      c_nrm = None
    else:
      c_nrm = normalize(inputs.c, shift=stats['c']['mean'], scale=stats['c']['std'])
    x_inp_nrm = 2 * ((inputs.x_inp - stats['x']['min']) / (stats['x']['max'] - stats['x']['min'])) - 1
    x_out_nrm = 2 * ((inputs.x_out - stats['x']['min']) / (stats['x']['max'] - stats['x']['min'])) - 1
    if inputs.t is None:
      t_nrm = None
    else:
      t_nrm = (inputs.t - stats['t']['min']) / (stats['t']['max'] - stats['t']['min'])
    if inputs.tau is None:
      tau_nrm = None
    else:
      tau_nrm = (inputs.tau) / (stats['t']['max'] - stats['t']['min'])

    inputs_nrm = Inputs(
      u=u_nrm,
      c=c_nrm,
      x_inp=x_inp_nrm,
      x_out=x_out_nrm,
      t=t_nrm,
      tau=tau_nrm,
    )

    return inputs_nrm

  @abstractmethod
  def apply(self,
    variables,
    stats,
    inputs: Inputs,
    **kwargs,
  ):
    """Normalizes raw inputs and applies the operator on it."""
    pass

  def unroll(self,
    variables,
    stats,
    num_steps: int,
    inputs: Inputs,
    **kwargs,
  ):
    """Apply the stepper num_steps times."""
    # NOTE: Assuming constant x in time
    # NOTE: Assuming constant c in time

    def scan_fn_fractional(carry, forcing):
      u_inp, t_inp = carry
      tau = forcing
      _inputs = Inputs(
        u=u_inp,
        c=inputs.c,
        x_inp=inputs.x_inp,
        x_out=inputs.x_out,
        t=t_inp,
        tau=tau
      )
      u_out = self.apply(
        variables,
        stats,
        inputs=_inputs,
        **kwargs,
      )
      u_next = u_out
      t_next = t_inp + tau
      carry = (u_next, t_next)
      return carry, u_out

    # Split tau in num_steps fractional parts
    tau_tiled = jnp.repeat(inputs.tau, repeats=num_steps)
    tau_fract = tau_tiled / num_steps
    forcing = tau_fract

    (u_out, _), _ = jax.lax.scan(f=scan_fn_fractional,
      init=(inputs.u, inputs.t), xs=forcing, length=num_steps)

    return u_out

  @abstractmethod
  def get_loss_inputs(self,
    variables,
    stats,
    inputs: Inputs,
    **kwargs,
  ):
    """Calculates prediction and target variables, ready to be given as input to the loss function."""
    pass

  def get_intermediates(self,
    variables,
    stats,
    inputs: Inputs,
    **kwargs,
  ):
    """Gets the intermediate values that are computed in the model."""

    # Normalize inputs
    inputs_nrm = self.normalize_inputs(stats, inputs)

    # Get predicted normalized derivatives
    _, state = self._apply_operator(
      variables,
      inputs=inputs_nrm,
      capture_intermediates=True,
      **kwargs,
    )

    return state['intermediates']

class TimeDerivativeStepper(Stepper):
  """
  Interprets the output of the model as the truncated Taylor approximation
  for the first derivative of the target function in time.
  """

  def apply(self,
    variables,
    stats,
    inputs: Inputs,
    **kwargs,
  ):
    """Normalizes raw inputs and applies the operator on it."""

    # Normalize inputs
    inputs_nrm = self.normalize_inputs(stats, inputs)

    # Get predicted normalized derivatives
    d_prd_nrm = self._apply_operator(
      variables,
      inputs=inputs_nrm,
      **kwargs,
    )

    # Unnormalize predicted derivatives
    d_prd = unnormalize(
      d_prd_nrm,
      mean=stats['der']['mean'],
      std=stats['der']['std'],
    )

    # Get predicted output
    u_prd = inputs.u + (d_prd * inputs.tau)

    return u_prd

  def get_loss_inputs(self,
    variables,
    stats,
    u_tgt: Array,
    inputs: Inputs,
    **kwargs,
  ):
    """Calculates prediction and target variables, ready to be given as input to the loss function."""

    # Normalize inputs
    inputs_nrm = self.normalize_inputs(stats, inputs)

    # Get predicted normalized derivatives
    d_prd_nrm = self._apply_operator(
      variables,
      inputs=inputs_nrm,
      **kwargs,
    )

    # Get target normalized derivatives
    d_tgt = (u_tgt - inputs.u) / inputs.tau
    d_tgt_nrm = normalize(
      d_tgt,
      shift=stats['der']['mean'],
      scale=stats['der']['std'],
    )

    return (d_tgt_nrm, d_prd_nrm)

class ResidualStepper(Stepper):
  """Assumes the output of the model to be the residual of the target function."""

  def apply(self,
    variables,
    stats,
    inputs: Inputs,
    **kwargs,
  ):
    """Normalizes raw inputs and applies the operator on it."""

    # Normalize inputs
    inputs_nrm = self.normalize_inputs(stats, inputs)

    # Get predicted normalized derivative
    r_prd_nrm = self._apply_operator(
      variables,
      inputs=inputs_nrm,
      **kwargs,
    )

    # Unnormalize predicted residuals
    r_prd = unnormalize(
      r_prd_nrm,
      mean=stats['res']['mean'],
      std=stats['res']['std'],
    )

    # Get predicted output
    u_prd = inputs.u + r_prd

    return u_prd

  def get_loss_inputs(self,
    variables,
    stats,
    u_tgt: Array,
    inputs: Inputs,
    **kwargs,
  ):
    """Calculates prediction and target variables, ready to be given as input to the loss function."""

    # Normalize inputs
    inputs_nrm = self.normalize_inputs(stats, inputs)

    # Get predicted normalized residuals
    r_prd_nrm = self._apply_operator(
      variables,
      inputs=inputs_nrm,
      **kwargs,
    )

    # Get target normalized residuals
    r_tgt = u_tgt - inputs.u
    r_tgt_nrm = normalize(
      r_tgt,
      shift=stats['res']['mean'],
      scale=stats['res']['std'],
    )

    return (r_tgt_nrm, r_prd_nrm)

class OutputStepper(Stepper):
  """Assumes the output of the model to be the actual values of the target function."""

  def apply(self,
    variables,
    stats,
    inputs: Inputs,
    **kwargs,
  ):
    """Normalizes raw inputs and applies the operator on it."""

    # Normalize inputs
    inputs_nrm = self.normalize_inputs(stats, inputs)

    # Get predicted normalized output
    u_prd_nrm = self._apply_operator(
      variables,
      inputs=inputs_nrm,
      **kwargs,
    )

    # Unnormalize predicted output
    u_prd = unnormalize(
      u_prd_nrm,
      mean=stats['u']['mean'],
      std=stats['u']['std'],
    )

    return u_prd

  def get_loss_inputs(self,
    variables,
    stats,
    u_tgt: Array,
    inputs: Inputs,
    **kwargs,
  ):
    """Calculates prediction and target variables, ready to be given as input to the loss function."""

    # Normalize inputs
    inputs_nrm = self.normalize_inputs(stats, inputs)

    # Get predicted normalized output
    u_prd_nrm = self._apply_operator(
      variables,
      inputs=inputs_nrm,
      **kwargs,
    )

    # Get target normalized output
    u_tgt_nrm = normalize(
      u_tgt,
      shift=stats['u']['mean'],
      scale=stats['u']['std'],
    )

    return (u_tgt_nrm, u_prd_nrm)

class AutoregressiveStepper:
  """Class for applying a time-stepping scheme autoregressively."""

  def __init__(self, stepper: Stepper, dt: float, tau_max: Union[None, float] = None):
    """
    Class for autoregressive Inference of an operator.

    Args:
        stepper: Uses an operator with proper stepping method.
        dt: Time resolution of the trajectory.
        tau_max: Maximum time difference of direct predictions. Defaults to None.
    """

    if tau_max is None:
      tau_max = dt
    self.dt = dt
    if tau_max >= dt:
      assert is_multiple(tau_max, dt)
      self.num_steps_direct = int(tau_max / dt)
      self._apply_operator = jax.checkpoint(stepper.apply)
    else:
      assert is_multiple(dt, tau_max)
      self.num_steps_direct = 1
      num_unrolls_per_step = int(dt / tau_max)
      def _stepper_unroll(*args, **kwargs):
        return stepper.unroll(*args, **kwargs, num_steps=num_unrolls_per_step)
      self._apply_operator = jax.checkpoint(_stepper_unroll)

  def unroll(self,
    variables: flax.typing.VariableDict,
    stats: flax.typing.VariableDict,
    num_steps: int,
    inputs: Inputs,
    key: flax.typing.PRNGKey = None,
    **kwargs,
  ) -> Array:
    """Unroll by taking all the necessary time steps. To be used when the whole trajectory is needed."""

    # NOTE: Assuming constant x in time
    # NOTE: Assuming constant c in time

    assert inputs.tau is None
    u_inp = inputs.u
    batch_size = u_inp.shape[0]
    num_pnodes = u_inp.shape[2]
    num_vars = u_inp.shape[3]
    t_inp = inputs.t.astype(float)
    random = (key is not None)
    if not random:
      key, _ = jax.random.split(jax.random.PRNGKey(0))  # NOTE: It won't be used

    def scan_fn_direct(carry, forcing):
      u_inp, t_inp = carry
      tau = forcing[0]
      key = forcing[-2:].astype('uint32')
      _inputs = Inputs(
        u=u_inp,
        c=inputs.c,
        x_inp=inputs.x_inp,
        x_out=inputs.x_out,
        t=t_inp,
        tau=tau,
      )
      u_out = self._apply_operator(
        variables,
        stats,
        inputs=_inputs,
        key=(key if random else None),
        **kwargs,
      )
      carry = (u_inp, t_inp)  # NOTE: The input is the same for all tau
      return carry, u_out

    def scan_fn_autoregressive(carry, forcing):
      u_inp, t_inp = carry
      tau = forcing[:-2]
      key = forcing[-2:].astype('uint32')
      _num_direct_steps = tau.shape[0]
      keys = jax.random.split(key, num=_num_direct_steps)
      forcing = jnp.concatenate([tau.reshape(-1, 1), keys], axis=-1)
      _, u_out = jax.lax.scan(f=scan_fn_direct,
        init=(u_inp, t_inp), xs=forcing, length=_num_direct_steps)
      u_out = jnp.squeeze(u_out, axis=2).swapaxes(0, 1)
      u_next = u_out[:, -1:]
      t_next = t_inp + self.dt * self.num_steps_direct
      carry = (u_next, t_next)
      return carry, u_out

    # Get full sets of direct predictions
    num_jumps = num_steps // self.num_steps_direct
    tau_tiled = self.dt * jnp.tile(
      jnp.arange(1, self.num_steps_direct+1), reps=(num_jumps, 1)
    ).reshape(num_jumps, self.num_steps_direct)
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=num_jumps)
    forcings = jnp.concatenate([tau_tiled, keys], axis=-1)
    (u_next, t_next), rollout_full = jax.lax.scan(
      f=scan_fn_autoregressive,
      init=(u_inp, t_inp),
      xs=forcings,
      length=num_jumps,
    )
    rollout_full = rollout_full.swapaxes(0, 1)
    rollout_full = rollout_full.reshape(
      batch_size, (num_jumps*self.num_steps_direct), num_pnodes, num_vars)
    rollout = jnp.concatenate([u_inp, rollout_full], axis=1)

    # Get the last set of direct predictions partially (if necessary)
    num_steps_rem = num_steps % self.num_steps_direct
    if num_steps_rem:
      tau_tiled = self.dt * jnp.arange(1, num_steps_rem+1).reshape(1, num_steps_rem)
      key, subkey = jax.random.split(key, num=2)
      keys = subkey.reshape(1, 2)
      forcings = jnp.concatenate([tau_tiled, keys], axis=-1)
      (u_next, t_next), rollout_part = jax.lax.scan(
        f=scan_fn_autoregressive,
        init=(u_next, t_next),
        xs=forcings,
        length=1
      )
      rollout_part = rollout_part.swapaxes(0, 1)
      rollout_part = rollout_part.reshape(
        batch_size, num_steps_rem, num_pnodes, num_vars)
      rollout = jnp.concatenate([rollout, rollout_part], axis=1)

    # Exclude the last timestep because it is returned separately
    rollout = rollout[:, :-1]

    return rollout, u_next

  def jump(self,
    variables: flax.typing.VariableDict,
    stats: flax.typing.VariableDict,
    num_jumps: int,
    inputs: Inputs,
    key: flax.typing.PRNGKey = None,
    **kwargs,
  ) -> Array:
    """Takes num_jumps large steps, each of length num_steps_direct. To be used when only the last time step is needed."""

    assert inputs.tau is None
    u_inp = inputs.u
    t_inp = inputs.t.astype(float)
    random = (key is not None)
    if not random:
      key, _ = jax.random.split(jax.random.PRNGKey(0))  # Won't be used

    def scan_fn(carry, forcing):
      u_inp, t_inp = carry
      subkey = forcing if random else None
      tau = self.dt * self.num_steps_direct
      _inputs = Inputs(
        u=u_inp,
        c=inputs.c,
        x_inp=inputs.x_inp,
        x_out=inputs.x_out,
        t=t_inp,
        tau=tau,
      )
      u_out = self._apply_operator(
        variables,
        stats,
        inputs=_inputs,
        key=subkey,
        **kwargs,
      )
      u_inp_next = u_out
      t_inp_next = t_inp + tau
      carry = (u_inp_next, t_inp_next)
      rollout = None
      return carry, rollout

    keys = jax.random.split(key, num=num_jumps)
    forcings = keys
    (u_next, t_next), _ = jax.lax.scan(
      f=scan_fn,
      init=(u_inp, t_inp),
      xs=forcings,
      length=num_jumps,
    )

    return u_next
