from datetime import datetime
import functools
import json
import pickle
from time import time
from typing import Tuple, Any, Mapping, Iterable, Callable, Union

from absl import app, flags, logging
from flax.jax_utils import replicate, unreplicate
import flax.typing
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from flax.training.common_utils import shard, shard_prng_key
import jax
import jax.numpy as jnp
import jax.tree_util as tree
from jax.tree_util import PyTreeDef
import numpy as np
import optax
import orbax.checkpoint
from matplotlib import pyplot as plt

from rigno.dataset import Dataset, Batch
from rigno.experiments import DIR_EXPERIMENTS
from rigno.metrics import BatchMetrics, Metrics, EvalMetrics
from rigno.metrics import mse_loss
from rigno.metrics import mse_error, rel_lp_error_norm
from rigno.metrics import normalized_rel_lp_error_mean
from rigno.models.operator import AbstractOperator, Inputs
from rigno.models.rigno import RIGNO
from rigno.models.rigno import RegionInteractionGraphBuilder
from rigno.stepping import AutoregressiveStepper
from rigno.stepping import TimeDerivativeStepper, ResidualStepper, OutputStepper
from rigno.test import get_direct_estimations
from rigno.utils import disable_logging, Array, shuffle_arrays, split_arrays, is_multiple


NUM_DEVICES = jax.local_device_count()
EVAL_FREQ = 100
IDX_FN = 14

FLAGS = flags.FLAGS

def define_flags():
  # FLAGS::general
  flags.DEFINE_string(name='exp', default='000', required=False,
    help='Name of the experiment.'
  )
  flags.DEFINE_string(name='datetime', default=None, required=False,
    help='A string representing the current datetime.'
  )
  flags.DEFINE_string(name='datadir', default=None, required=True,
    help='Path of the folder containing the datasets.'
  )
  flags.DEFINE_string(name='datapath', default=None, required=True,
    help='Relative path inside the data directory. Example: "unstructured/ACE".'
  )
  flags.DEFINE_string(name='params', default=None, required=False,
    help='Path of the previous experiment containing the initial parameters.'
  )
  flags.DEFINE_integer(name='seed', default=44, required=False,
    help='Seed for random number generator.'
  )
  flags.DEFINE_integer(name='time_downsample_factor', default=2, required=False,
    help='Factor for downsampling the time resolution.'
  )
  flags.DEFINE_float(name='space_downsample_factor', default=1., required=False,
    help='Factor for downsampling the space resolution.'
  )

  # FLAGS::training
  flags.DEFINE_integer(name='batch_size', default=2, required=False,
    help='Size of a batch of training samples. Must be equal or greater than number of GPUs.'
  )
  flags.DEFINE_integer(name='epochs', default=2000, required=False,
    help='Number of training epochs.'
  )
  flags.DEFINE_float(name='lr_init', default=1e-05, required=False,
    help='Initial learning rate in the onecycle scheduler.'
  )
  flags.DEFINE_float(name='lr_peak', default=2e-04, required=False,
    help='Peak learning rate in the onecycle scheduler.'
  )
  flags.DEFINE_float(name='lr_base', default=1e-05, required=False,
    help='Final learning rate in the onecycle scheduler.'
  )
  flags.DEFINE_float(name='lr_lowr', default=1e-06, required=False,
    help='Final learning rate in the exponential decay.'
  )
  flags.DEFINE_string(name='stepper', default='der', required=False,
    help='Type of the stepper. Set to "out" for time-independent datasets.'
  )
  flags.DEFINE_integer(name='tau_max', default=7, required=False,
    help='Maximum number of time steps between input/output pairs during training. Set to 0 for time-independent datasets.'
  )
  flags.DEFINE_boolean(name='fractional', default=False, required=False,
    help='If passed, fine-tunes with fractional pairing strategy.'
  )
  flags.DEFINE_integer(name='n_train', default=(2**9), required=False,
    help='Number of training samples.'
  )
  flags.DEFINE_integer(name='n_valid', default=(2**7), required=False,
    help='Number of validation samples.'
  )
  flags.DEFINE_integer(name='n_test', default=(2**8), required=False,
    help='Number of test samples.'
  )

  # FLAGS::model::RIGNO
  flags.DEFINE_float(name='mesh_subsample_factor', default=4.0, required=False,
    help='Factor for random subsampling of hierarchical meshes.'
  )
  flags.DEFINE_float(name='overlap_factor_p2r', default=1.0, required=False,
    help='Overlap factor for p2r edges (encoder).'
  )
  flags.DEFINE_float(name='overlap_factor_r2p', default=2.0, required=False,
    help='Overlap factor for r2p edges (decoder).'
  )
  flags.DEFINE_integer(name='rmesh_levels', default=6, required=False,
    help='Number of multimesh connection levels (processor).'
  )
  flags.DEFINE_integer(name='node_coordinate_freqs', default=4, required=False,
    help='Number of frequencies for encoding periodic node coordinates.'
  )
  flags.DEFINE_integer(name='node_latent_size', default=128, required=False,
    help='Size of latent node features.'
  )
  flags.DEFINE_integer(name='edge_latent_size', default=128, required=False,
    help='Size of latent edge features.'
  )
  flags.DEFINE_integer(name='mlp_hidden_layers', default=1, required=False,
    help='Number of hidden layers of all MLPs.'
  )
  flags.DEFINE_integer(name='processor_steps', default=12, required=False,
    help='Number of message-passing steps in the processor.'
  )
  flags.DEFINE_float(name='p_edge_masking', default=0.5, required=False,
    help='Probability of random edge masking.'
  )

def train(
  key: flax.typing.PRNGKey,
  model: AbstractOperator,
  state: TrainState,
  dataset: Dataset,
  graph_builder: RegionInteractionGraphBuilder,
  tau_max: int,
  unroll: bool,
  epochs: int,
  epochs_before: int = 0,
  loss_fn: Callable = mse_loss,
) -> TrainState:
  """Trains a model and returns the model state."""

  # Set constants
  num_samples_trn = dataset.nums['train']
  num_times = dataset.shape[1]
  num_pnodes = dataset.shape[2]
  num_vars = dataset.shape[3]
  assert num_samples_trn % FLAGS.batch_size == 0
  num_batches = num_samples_trn // FLAGS.batch_size
  assert FLAGS.batch_size % NUM_DEVICES == 0
  batch_size_per_device = FLAGS.batch_size // NUM_DEVICES
  evaluation_frequency = (
    (FLAGS.epochs // EVAL_FREQ) if (FLAGS.epochs >= EVAL_FREQ)
    else 1
  )

  # Store the initial time
  time_int_pre = time()

  # Define the permissible lead times
  if dataset.time_dependent:
    num_init_times = num_times - 1
    assert num_init_times > 0
    assert tau_max < num_times
    num_init_times_full = max(0, num_times - tau_max)
    num_init_times_part = num_init_times - num_init_times_full
    num_valid_pairs = (
      num_init_times_full * tau_max
      + (num_init_times_part * (num_init_times_part+1) // 2)
    )
    init_times = jnp.arange(num_times - 1)

  # Define the steppers
  if FLAGS.stepper == 'der':
    stepper = TimeDerivativeStepper(operator=model)
  elif FLAGS.stepper == 'res':
    stepper = ResidualStepper(operator=model)
  elif FLAGS.stepper == 'out':
    stepper = OutputStepper(operator=model)
  else:
    raise ValueError
  if dataset.time_dependent:
    autoregressive = AutoregressiveStepper(stepper=stepper, dt=dataset.dt)

  # Set the normalization statistics
  stats = {
    key: {
      k: (jnp.array(v) if (v is not None) else None)
      for k, v in val.items()
    }
    for key, val in dataset.stats.items()
  }

  # Replicate state, stats, and graphs
  # NOTE: Internally uses jax.device_put_replicate
  state = replicate(state)
  stats = replicate(stats)

  @functools.partial(jax.pmap, axis_name='device')
  def _train_one_batch(
    key: flax.typing.PRNGKey,
    state: TrainState,
    stats: dict,
    batch: Batch,
  ) -> Tuple[TrainState, Array, Array]:
    """Loads a batch, normalizes it, updates the state based on it, and returns it."""

    def _update_state_per_subbatch(
      key: flax.typing.PRNGKey,
      state: TrainState,
      u_inp: Array,
      c_inp: Array,
      x_inp: Array,
      t_inp: Array,
      tau: Array,
      u_tgt: Array,
      x_out: Array,
    ) -> Tuple[TrainState, Array, PyTreeDef]:
      # NOTE: INPUT SHAPES [batch_size_per_device, ...]

      def _get_loss_and_grads(
        key: flax.typing.PRNGKey,
        params: flax.typing.Collection,
        u_inp: Array,
        c_inp: Array,
        x_inp: Array,
        t_inp: Array,
        tau: Array,
        u_tgt: Array,
        x_out: Array,
      ) -> Tuple[Array, PyTreeDef]:
        """
        Computes the loss and the gradients of the loss w.r.t the parameters.
        """

        def _compute_loss(
          params: flax.typing.Collection,
          u_inp: Array,
          c_inp: Array,
          x_inp: Array,
          t_inp: Array,
          tau: Array,
          u_tgt: Array,
          x_out: Array,
          key: flax.typing.PRNGKey,
        ) -> Array:
          """Computes the prediction of the model and returns its loss."""

          variables = {'params': params}

          # Get the output
          key, subkey = jax.random.split(key)
          inputs = Inputs(
            u=u_inp,
            c=c_inp,
            x_inp=x_inp,
            x_out=x_out,
            t=t_inp,
            tau=tau,
          )
          _loss_inputs = stepper.get_loss_inputs(
            variables=variables,
            stats=stats,
            u_tgt=u_tgt,
            inputs=inputs,
            graphs=graph_builder.build_graphs(batch.g),
            key=subkey,
          )

          return loss_fn(*_loss_inputs)

        # Split tau for unrolling
        if unroll:
          # Choose a random split fraction
          key, subkey = jax.random.split(key)
          frac = jax.random.uniform(key=subkey, shape=tau.shape)
          # Flip a coin to decide about splitting
          if isinstance(stepper, TimeDerivativeStepper):
            # NOTE: Only applicable with this implementation for derivative stepping
            key, subkey = jax.random.split(key)
            P = .5
            coin = jax.random.uniform(key=subkey, shape=tau.shape) > P
            frac = frac * coin
          # Calculate the approximation tau
          tau_cutoff_approximation = 1 * dataset.dt  # minimum approximation fractional tau
          if isinstance(stepper, TimeDerivativeStepper):
            tau_cutoff_inference = .2 * dataset.dt  # minimum inference fractional tau
          else:
            tau_cutoff_inference = .0 * dataset.dt  # minimum inference fractional tau
          tau_splittable_part = jnp.clip((tau - tau_cutoff_approximation - tau_cutoff_inference), 0, None)
          tau_inf = tau_cutoff_approximation + frac * tau_splittable_part
          # Avoid inference with tau=0 if necessary
          if isinstance(stepper, TimeDerivativeStepper):
            # NOTE: approximation with tau=0 works fine with derivative stepping
            # NOTE: inference with tau=0 does not work with derivative stepping
            tau_inf = jnp.where(jnp.isclose(tau_inf, tau_cutoff_approximation), 0., tau_inf)
          # Get intermediary output by inferring the model
          key, subkey = jax.random.split(key)
          inputs = Inputs(
            u=u_inp,
            c=c_inp,
            x_inp=x_inp,
            x_out=x_out,
            t=t_inp,
            tau=tau_inf,
          )
          u_int = stepper.apply(
            variables={'params': params},
            stats=stats,
            inputs=inputs,
            graphs=graph_builder.build_graphs(batch.g),
            key=subkey,
          )
          c_int = c_inp
          x_int = x_inp
          t_int = t_inp + tau_inf
          tau_trn = tau - tau_inf
        else:
          u_int = u_inp
          c_int = c_inp
          x_int = x_inp
          t_int = t_inp
          tau_trn = tau

        # Compute gradients
        key, subkey = jax.random.split(key)
        loss, grads = jax.value_and_grad(_compute_loss)(
          params, u_int, c_int, x_int, t_int, tau_trn, u_tgt, x_out, key=subkey)

        return loss, grads

      # Update state, loss, and gradients
      _loss, _grads = _get_loss_and_grads(
        key=key,
        params=state.params,
        u_inp=u_inp,
        c_inp=c_inp,
        x_inp=x_inp,
        t_inp=t_inp,
        tau=tau,
        u_tgt=u_tgt,
        x_out=x_out,
      )
      # Synchronize loss and gradients
      loss = jax.lax.pmean(_loss, axis_name='device')
      grads = jax.lax.pmean(_grads, axis_name='device')
      # Apply gradients
      state = state.apply_gradients(grads=grads)

      return state, loss, grads

    if dataset.time_dependent:
      # Index trajectories and times and collect input/output pairs
      # -> [num_init_times, batch_size_per_device, ...]
      u_inp_batch = jax.vmap(
          lambda lt: jax.lax.dynamic_slice_in_dim(
            operand=batch.u,
            start_index=(lt), slice_size=1, axis=1)
      )(init_times)
      c_inp_batch = jax.vmap(
          lambda lt: jax.lax.dynamic_slice_in_dim(
            operand=batch.c,
            start_index=(lt), slice_size=1, axis=1)
      )(init_times) if (batch.c is not None) else None
      x_inp_batch = jax.vmap(
          lambda lt: jax.lax.dynamic_slice_in_dim(
            operand=batch.x,
            start_index=(lt), slice_size=1, axis=1)
      )(init_times)
      t_inp_batch = jax.vmap(
          lambda lt: jax.lax.dynamic_slice_in_dim(
            operand=batch.t,
            start_index=(lt), slice_size=1, axis=1)
      )(init_times)
      u_tgt_batch = jax.vmap(
          lambda lt: jax.lax.dynamic_slice_in_dim(
            operand=jnp.concatenate([batch.u, jnp.zeros_like(batch.u)], axis=1),
            start_index=(lt+1), slice_size=tau_max, axis=1)
      )(init_times)
      t_tgt_batch = jax.vmap(
          lambda lt: jax.lax.dynamic_slice_in_dim(
            operand=jnp.concatenate([batch.t, jnp.zeros_like(batch.t)], axis=1),
            start_index=(lt+1), slice_size=tau_max, axis=1)
      )(init_times)
      x_out_batch = jax.vmap(
          lambda lt: jax.lax.dynamic_slice_in_dim(
            operand=jnp.concatenate([batch.x, jnp.zeros_like(batch.x)], axis=1),
            start_index=(lt+1), slice_size=tau_max, axis=1)
      )(init_times)

      # Repeat inputs along the time axis to match with u_tgt
      # -> [num_init_times, batch_size_per_device, tau_max, ...]
      u_inp_batch = jnp.tile(u_inp_batch, reps=(1, 1, tau_max, 1, 1))
      c_inp_batch = jnp.tile(c_inp_batch, reps=(1, 1, tau_max, 1, 1)) if (batch.c is not None) else None
      x_inp_batch = jnp.tile(x_inp_batch, reps=(1, 1, tau_max, 1, 1))
      t_inp_batch = jnp.tile(t_inp_batch, reps=(1, 1, tau_max, 1, 1))

      # Put all pairs along the batch axis
      # -> [batch_size_per_device * num_init_times * tau_max, ...]
      u_inp_batch = u_inp_batch.reshape((num_init_times*batch_size_per_device*tau_max), 1, num_pnodes, -1)
      x_inp_batch = x_inp_batch.reshape((num_init_times*batch_size_per_device*tau_max), 1, num_pnodes, -1)
      c_inp_batch = c_inp_batch.reshape(
        (num_init_times*batch_size_per_device*tau_max), 1, num_pnodes, -1) if (batch.c is not None) else None
      t_inp_batch = t_inp_batch.reshape((num_init_times*batch_size_per_device*tau_max), 1, 1, 1)
      t_tgt_batch = t_tgt_batch.reshape((num_init_times*batch_size_per_device*tau_max), 1, 1, 1)
      u_tgt_batch = u_tgt_batch.reshape((num_init_times*batch_size_per_device*tau_max), 1, num_pnodes, -1)
      x_out_batch = x_out_batch.reshape((num_init_times*batch_size_per_device*tau_max), 1, num_pnodes, -1)

      # Get tau as the difference between input and target t
      tau_batch = t_tgt_batch - t_inp_batch

      # Remove the invalid pairs
      # -> [batch_size_per_device * num_valid_pairs, ...]
      offset_full_init_times = (num_times - tau_max) * tau_max * batch_size_per_device
      idx_invalid_pairs = np.array([
        (offset_full_init_times + (_d * batch_size_per_device + _b) * tau_max - (_n + 1))
        for _d in range(tau_max - 1)
        for _b in range(1, batch_size_per_device + 1)
        for _n in range(_d + 1)
      ]).astype(int)
      u_inp_batch = jnp.delete(u_inp_batch, idx_invalid_pairs, axis=0)
      c_inp_batch = jnp.delete(c_inp_batch, idx_invalid_pairs, axis=0) if (batch.c is not None) else None
      x_inp_batch = jnp.delete(x_inp_batch, idx_invalid_pairs, axis=0)
      t_inp_batch = jnp.delete(t_inp_batch, idx_invalid_pairs, axis=0)
      tau_batch = jnp.delete(tau_batch, idx_invalid_pairs, axis=0)
      u_tgt_batch = jnp.delete(u_tgt_batch, idx_invalid_pairs, axis=0)
      x_out_batch = jnp.delete(x_out_batch, idx_invalid_pairs, axis=0)

      # Shuffle and split the pairs
      # -> [num_valid_pairs, batch_size_per_device, ...]
      num_valid_pairs = u_tgt_batch.shape[0] // batch_size_per_device
      key, subkey = jax.random.split(key)
      if batch.c is None:
        u_inp_batch, x_inp_batch, t_inp_batch, tau_batch, u_tgt_batch, x_out_batch = shuffle_arrays(
          subkey, [u_inp_batch, x_inp_batch, t_inp_batch, tau_batch, u_tgt_batch, x_out_batch])
        u_inp_batch, x_inp_batch, t_inp_batch, tau_batch, u_tgt_batch, x_out_batch = split_arrays(
          [u_inp_batch, x_inp_batch, t_inp_batch, tau_batch, u_tgt_batch, x_out_batch], size=batch_size_per_device)
      else:
        u_inp_batch, c_inp_batch, x_inp_batch, t_inp_batch, tau_batch, u_tgt_batch, x_out_batch = shuffle_arrays(
          subkey, [u_inp_batch, c_inp_batch, x_inp_batch, t_inp_batch, tau_batch, u_tgt_batch, x_out_batch])
        u_inp_batch, c_inp_batch, x_inp_batch, t_inp_batch, tau_batch, u_tgt_batch, x_out_batch = split_arrays(
          [u_inp_batch, c_inp_batch, x_inp_batch, t_inp_batch, tau_batch, u_tgt_batch, x_out_batch], size=batch_size_per_device)
    else:
      num_valid_pairs = 1
      # Prepare time-independent input-output pairs
      # -> [1, batch_size_per_device, ...]
      u_inp_batch = jnp.expand_dims(batch.c, axis=0)
      x_inp_batch = jnp.expand_dims(batch.x, axis=0)
      c_inp_batch = None
      t_inp_batch = None
      tau_batch = None
      u_tgt_batch = jnp.expand_dims(batch.u, axis=0)
      x_out_batch = jnp.expand_dims(batch.x, axis=0)

    # Add loss and gradients for each subbatch
    def _update_state(i, carry):
      # Update state, loss, and gradients
      _state, _loss_carried, _grads_carried, _key_carried = carry
      _key_updated, _subkey = jax.random.split(_key_carried)
      _state, _loss_subbatch, _grads_subbatch = _update_state_per_subbatch(
        key=_subkey,
        state=_state,
        u_inp=u_inp_batch[i],
        c_inp=(c_inp_batch[i] if (c_inp_batch is not None) else None),
        x_inp=x_inp_batch[i],
        t_inp=(t_inp_batch[i] if (t_inp_batch is not None) else None),
        tau=(tau_batch[i] if (tau_batch is not None) else None),
        u_tgt=u_tgt_batch[i],
        x_out=x_out_batch[i],
      )
      # Update the carried loss and gradients of the subbatch
      _loss_updated = _loss_carried + _loss_subbatch / num_valid_pairs
      _grads_updated = jax.tree_util.tree_map(
        lambda g_old, g_new: (g_old + g_new / num_valid_pairs),
        _grads_carried,
        _grads_subbatch,
      )

      return _state, _loss_updated, _grads_updated, _key_updated

    # Loop over the pairs
    _init_state = state
    _init_loss = 0.
    _init_grads = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), state.params)
    key, _init_key = jax.random.split(key)
    state, loss, grads, _ = jax.lax.fori_loop(
      lower=0,
      upper=num_valid_pairs,
      body_fun=_update_state,
      init_val=(_init_state, _init_loss, _init_grads, _init_key)
    )

    return state, loss, grads

  def train_one_epoch(
    key: flax.typing.PRNGKey,
    state: TrainState,
    batches: Iterable[Batch],
  ) -> Tuple[TrainState, Array, Array]:
    """Updates the state based on accumulated losses and gradients."""

    # Loop over the batches
    loss_epoch = 0.
    grad_epoch = 0.
    for batch in batches:

      # Split the batch between devices
      # -> [NUM_DEVICES, batch_size_per_device, ...]
      batch = Batch(
        u=shard(batch.u),
        c=shard(batch.c),
        x=shard(batch.x),
        t=shard(batch.t),
        g=shard(batch.g),
      )

      # Get loss and updated state
      subkey, key = jax.random.split(key)
      subkey = shard_prng_key(subkey)
      state, loss, grads = _train_one_batch(subkey, state, stats, batch)
      # NOTE: Using the first element of replicated loss and grads
      loss_epoch += loss[0] * FLAGS.batch_size / num_samples_trn
      grad_epoch += np.mean(jax.tree_util.tree_flatten(
        jax.tree_util.tree_map(jnp.mean, jax.tree_util.tree_map(lambda g: jnp.abs(g[0]), grads)))[0]) / num_batches

    return state, loss_epoch, grad_epoch

  @functools.partial(jax.pmap, static_broadcasted_argnums=(0,))
  def _evaluate_direct_prediction(
    tau_ratio: Union[None, float, int],
    state: TrainState,
    stats,
    batch: Batch
  ) -> Mapping:
    """
    Evaluates the one-step predictions of the model by giving all the snapshots as input and
    comparing the results with the next snapshot. The input dataset must be raw (not normalized).
    """

    if tau_ratio < 1:
      assert is_multiple(1., tau_ratio)
      step = lambda *args, **kwargs: stepper.unroll(*args, **kwargs, num_steps=int(1 / tau_ratio))
      _tau_ratio = 1
    else:
      assert isinstance(tau_ratio, int)
      step = stepper.apply
      _tau_ratio = tau_ratio

    u_prd = get_direct_estimations(
      step=step,
      graph_builder=graph_builder,
      variables={'params': state.params},
      stats=stats,
      batch=batch,
      tau=(_tau_ratio * dataset.dt),
    )

    # Get mean errors per each sample in the batch
    _u_pair = (batch.u[:, _tau_ratio:], u_prd[:, :-_tau_ratio])
    batch_metrics = BatchMetrics(
      mse=mse_error(*_u_pair),
      l1=rel_lp_error_norm(*_u_pair, p=1),
      l2=rel_lp_error_norm(*_u_pair, p=2),
      _l1=normalized_rel_lp_error_mean(*_u_pair, dataset.metadata, p=1),
      _l2=normalized_rel_lp_error_mean(*_u_pair, dataset.metadata, p=2),
    )

    return batch_metrics.__dict__

  @jax.pmap
  def _evaluate_rollout_prediction(
    state: TrainState,
    stats,
    batch: Batch
  ) -> Mapping:
    """
    Predicts the trajectories autoregressively and evaluates them.
    The input dataset must be raw (not normalized).
    """

    # Set inputs and target
    u_tgt = batch.u
    inputs = Inputs(
      u=batch.u[:, :1],
      c=(batch.c[:, :1] if (batch.c is not None) else None),
      x_inp=batch.x,
      x_out=batch.x,
      t=batch.t[:, :1],
      tau=None,
    )

    # Get unrolled predictions
    variables = {'params': state.params}
    _predictor = autoregressive
    u_prd, _ = _predictor.unroll(
      variables=variables,
      stats=stats,
      num_steps=num_times,
      inputs=inputs,
      graphs=graph_builder.build_graphs(batch.g),
    )

    # Calculate the errors
    batch_metrics = BatchMetrics(
      mse=mse_error(u_tgt, u_prd),
      l1=rel_lp_error_norm(u_tgt, u_prd, p=1),
      l2=rel_lp_error_norm(u_tgt, u_prd, p=2),
      _l1=normalized_rel_lp_error_mean(u_tgt, u_prd, dataset.metadata, p=1),
      _l2=normalized_rel_lp_error_mean(u_tgt, u_prd, dataset.metadata, p=2),
    )

    return batch_metrics.__dict__

  @jax.pmap
  def _evaluate_final_prediction(
    state: TrainState,
    stats,
    batch: Batch
  ) -> Mapping:
    """
    Evaluates the predictions at the final time only.
    The input dataset must be raw (not normalized).
    """

    if dataset.time_dependent:
      # Set input and target
      idx_fn = IDX_FN // FLAGS.time_downsample_factor
      u_tgt = batch.u[:, (idx_fn):(idx_fn+1)]
      inputs = Inputs(
        u=batch.u[:, :1],
        c=(batch.c[:, :1] if (batch.c is not None) else None),
        x_inp=batch.x,
        x_out=batch.x,
        t=batch.t[:, :1],
        tau=None,
      )

      # Get prediction at the final step
      _predictor = autoregressive
      _num_jumps = idx_fn // _predictor.num_steps_direct
      _num_direct_steps = idx_fn % _predictor.num_steps_direct
      variables = {'params': state.params}
      u_prd = _predictor.jump(
        variables=variables,
        stats=stats,
        num_jumps=_num_jumps,
        inputs=inputs,
        graphs=graph_builder.build_graphs(batch.g),
      )
      if _num_direct_steps:
        _num_dt_jumped = _num_jumps * _predictor.num_steps_direct
        inputs = Inputs(
          u=u_prd,
          c=(batch.c[:, [_num_dt_jumped]] if (batch.c is not None) else None),
          x_inp=batch.x,
          x_out=batch.x,
          t=(batch.t[:, :1] + _num_dt_jumped * _predictor.dt),
          tau=None,
        )
        _, u_prd = _predictor.unroll(
          variables=variables,
          stats=stats,
          num_steps=_num_direct_steps,
          inputs=inputs,
          graphs=graph_builder.build_graphs(batch.g),
        )

    else:
      u_tgt = batch.u[:, [0]]
      u_prd = stepper.apply(
        variables={'params': state.params},
        stats=stats,
        inputs=Inputs(
          u=batch.c[:, [0]],
          c=None,
          x_inp=batch.x,
          x_out=batch.x,
          t=None,
          tau=None,
        ),
        graphs=graph_builder.build_graphs(batch.g),
      )

    # Calculate the errors
    batch_metrics = BatchMetrics(
      mse=mse_error(u_tgt, u_prd),
      l1=rel_lp_error_norm(u_tgt, u_prd, p=1),
      l2=rel_lp_error_norm(u_tgt, u_prd, p=2),
      _l1=normalized_rel_lp_error_mean(u_tgt, u_prd, dataset.metadata, p=1),
      _l2=normalized_rel_lp_error_mean(u_tgt, u_prd, dataset.metadata, p=2),
    )

    return batch_metrics.__dict__

  def evaluate(
    state: TrainState,
    batches: Iterable[Batch],
    direct: bool = True,
    rollout: bool = False,
    final: bool = True,
  ) -> EvalMetrics:
    """Runs all the evaluations and returns the metrics."""

    metrics_direct_tau_frac: list[BatchMetrics] = []
    metrics_direct_tau_min: list[BatchMetrics] = []
    metrics_direct_tau_max: list[BatchMetrics] = []
    metrics_rollout: list[BatchMetrics] = []
    metrics_final: list[BatchMetrics] = []

    # Turn off unrelevent evaluations
    if not dataset.time_dependent:
      direct = False
      rollout = False

    for batch in batches:
      # Split the batch between devices
      # -> [NUM_DEVICES, batch_size_per_device, ...]
      batch = Batch(
        u=shard(batch.u),
        c=shard(batch.c),
        x=shard(batch.x),
        t=shard(batch.t),
        g=shard(batch.g),
      )

      # Evaluate direct prediction
      if direct:
        # tau = .5*dt
        batch_metrics_direct_tau_frac = _evaluate_direct_prediction(.5, state, stats, batch)
        batch_metrics_direct_tau_frac = BatchMetrics(**batch_metrics_direct_tau_frac)
        # Re-arrange the sub-batches gotten from each device
        batch_metrics_direct_tau_frac.reshape(shape=(batch_size_per_device * NUM_DEVICES, 1))
        # Append the metrics to the list
        metrics_direct_tau_frac.append(batch_metrics_direct_tau_frac)

        # tau = dt
        batch_metrics_direct_tau_min = _evaluate_direct_prediction(1, state, stats, batch)
        batch_metrics_direct_tau_min = BatchMetrics(**batch_metrics_direct_tau_min)
        # Re-arrange the sub-batches gotten from each device
        batch_metrics_direct_tau_min.reshape(shape=(batch_size_per_device * NUM_DEVICES, 1))
        # Append the metrics to the list
        metrics_direct_tau_min.append(batch_metrics_direct_tau_min)

        # tau = tau_max
        batch_metrics_direct_tau_max = _evaluate_direct_prediction(FLAGS.tau_max, state, stats, batch)
        batch_metrics_direct_tau_max = BatchMetrics(**batch_metrics_direct_tau_max)
        # Re-arrange the sub-batches gotten from each device
        batch_metrics_direct_tau_max.reshape(shape=(batch_size_per_device * NUM_DEVICES, 1))
        # Append the metrics to the list
        metrics_direct_tau_max.append(batch_metrics_direct_tau_max)

      # Evaluate rollout prediction
      if rollout:
        batch_metrics_rollout = _evaluate_rollout_prediction(state, stats, batch)
        batch_metrics_rollout = BatchMetrics(**batch_metrics_rollout)
        # Re-arrange the sub-batches gotten from each device
        batch_metrics_rollout.reshape(shape=(batch_size_per_device * NUM_DEVICES, 1))
        # Compute and store metrics
        metrics_rollout.append(batch_metrics_rollout)

      # Evaluate final prediction
      if final:
        if dataset.time_dependent: assert (IDX_FN // FLAGS.time_downsample_factor) < batch.u.shape[2]
        batch_metrics_final = _evaluate_final_prediction(state, stats, batch)
        batch_metrics_final = BatchMetrics(**batch_metrics_final)
        # Re-arrange the sub-batches gotten from each device
        batch_metrics_final.reshape(shape=(batch_size_per_device * NUM_DEVICES, 1))
        # Append the errors to the list
        metrics_final.append(batch_metrics_final)

    # Aggregate over the batch dimension and compute norm per variable
    metrics_direct_tau_frac = Metrics(
      mse=jnp.median(jnp.concatenate([m.mse for m in metrics_direct_tau_frac]), axis=0).item(),
      l1=jnp.median(jnp.concatenate([m.l1 for m in metrics_direct_tau_frac]), axis=0).item(),
      l2=jnp.median(jnp.concatenate([m.l2 for m in metrics_direct_tau_frac]), axis=0).item(),
      _l1=jnp.median(jnp.concatenate([m._l1 for m in metrics_direct_tau_frac]), axis=0).item(),
      _l2=jnp.median(jnp.concatenate([m._l2 for m in metrics_direct_tau_frac]), axis=0).item(),
    ) if direct else None
    metrics_direct_tau_min = Metrics(
      mse=jnp.median(jnp.concatenate([m.mse for m in metrics_direct_tau_min]), axis=0).item(),
      l1=jnp.median(jnp.concatenate([m.l1 for m in metrics_direct_tau_min]), axis=0).item(),
      l2=jnp.median(jnp.concatenate([m.l2 for m in metrics_direct_tau_min]), axis=0).item(),
      _l1=jnp.median(jnp.concatenate([m._l1 for m in metrics_direct_tau_min]), axis=0).item(),
      _l2=jnp.median(jnp.concatenate([m._l2 for m in metrics_direct_tau_min]), axis=0).item(),
    ) if direct else None
    metrics_direct_tau_max = Metrics(
      mse=jnp.median(jnp.concatenate([m.mse for m in metrics_direct_tau_max]), axis=0).item(),
      l1=jnp.median(jnp.concatenate([m.l1 for m in metrics_direct_tau_max]), axis=0).item(),
      l2=jnp.median(jnp.concatenate([m.l2 for m in metrics_direct_tau_max]), axis=0).item(),
      _l1=jnp.median(jnp.concatenate([m._l1 for m in metrics_direct_tau_max]), axis=0).item(),
      _l2=jnp.median(jnp.concatenate([m._l2 for m in metrics_direct_tau_max]), axis=0).item(),
    ) if direct else None
    metrics_rollout = Metrics(
      mse=jnp.median(jnp.concatenate([m.mse for m in metrics_rollout]), axis=0).item(),
      l1=jnp.median(jnp.concatenate([m.l1 for m in metrics_rollout]), axis=0).item(),
      l2=jnp.median(jnp.concatenate([m.l2 for m in metrics_rollout]), axis=0).item(),
      _l1=jnp.median(jnp.concatenate([m._l1 for m in metrics_rollout]), axis=0).item(),
      _l2=jnp.median(jnp.concatenate([m._l2 for m in metrics_rollout]), axis=0).item(),
    ) if rollout else None
    metrics_final = Metrics(
      mse=jnp.median(jnp.concatenate([m.mse for m in metrics_final]), axis=0).item(),
      l1=jnp.median(jnp.concatenate([m.l1 for m in metrics_final]), axis=0).item(),
      l2=jnp.median(jnp.concatenate([m.l2 for m in metrics_final]), axis=0).item(),
      _l1=jnp.median(jnp.concatenate([m._l1 for m in metrics_final]), axis=0).item(),
      _l2=jnp.median(jnp.concatenate([m._l2 for m in metrics_final]), axis=0).item(),
    ) if final else None

    # Build the metrics object
    metrics = EvalMetrics(
      direct_tau_frac=(metrics_direct_tau_frac if direct else Metrics()),
      direct_tau_min=(metrics_direct_tau_min if direct else Metrics()),
      direct_tau_max=(metrics_direct_tau_max if direct else Metrics()),
      rollout=(metrics_rollout if rollout else Metrics()),
      final=(metrics_final if final else Metrics()),
    )

    return metrics

  # Evaluate before training
  metrics_trn = evaluate(
    state=state,
    batches=dataset.batches(mode='train', batch_size=FLAGS.batch_size),
  )
  metrics_val = evaluate(
    state=state,
    batches=dataset.batches(mode='valid', batch_size=FLAGS.batch_size),
  )

  # Report the initial evaluations
  time_tot_pre = time() - time_int_pre
  tdsf = FLAGS.time_downsample_factor
  lr = state.opt_state[-1].hyperparams["learning_rate"][0].item()
  logging.info('\t'.join([
    f'DRCT: {tau_max * tdsf : 02d}dt',
    f'EPCH: {epochs_before : 04d}/{FLAGS.epochs : 04d}',
    f'LR: {lr : .2e}',
    f'TIME: {time_tot_pre : 06.1f}s',
    f'GRAD: {0. : .2e}',
    f'LOSS: {0. : .2e}',
    f'DR-{.5 * tdsf}: {metrics_val.direct_tau_frac._l1 : .2%}' if metrics_val.direct_tau_frac._l1 else '',
    f'DR-{tdsf}: {metrics_val.direct_tau_min._l1 : .2%}' if metrics_val.direct_tau_min._l1 else '',
    f'DR-{FLAGS.tau_max * tdsf}: {metrics_val.direct_tau_max._l1 : .2%}' if metrics_val.direct_tau_max._l1 else '',
    f'FN: {metrics_val.final._l1 : .2%}' if metrics_val.final._l1 else '',
    f'TRN-DR-{tdsf}: {metrics_trn.direct_tau_min._l1 : .2%}' if metrics_trn.direct_tau_min._l1 else '',
    f'TRN-DR-{FLAGS.tau_max * tdsf}: {metrics_trn.direct_tau_max._l1 : .2%}' if metrics_trn.direct_tau_max._l1 else '',
    f'TRN-FN: {metrics_trn.final._l1 : .2%}' if metrics_trn.final._l1 else '',
  ]))

  # Set up the checkpoint manager
  DIR = DIR_EXPERIMENTS / f'E{FLAGS.exp}' / FLAGS.datapath / FLAGS.datetime
  with disable_logging(level=logging.FATAL):
    (DIR / 'metrics').mkdir(exist_ok=True)
    (DIR / 'metrics/plots').mkdir(exist_ok=True)
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpointer_options = orbax.checkpoint.CheckpointManagerOptions(
      max_to_keep=1,
      keep_period=None,
      best_fn=(lambda metrics: metrics['valid']['final']['_l1']),
      best_mode='min',
      create=True,)
    checkpointer_save_args = orbax_utils.save_args_from_target(target={'state': state})
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
      (DIR / 'checkpoints'), checkpointer, checkpointer_options)


  checkpointed_metrics = []
  for epoch in range(1, epochs+1):
    # Store the initial time
    time_int = time()

    # Re-construct the graphs with a new PRNG key
    # NOTE: In order to prevent always training with the same rmesh nodes
    key, subkey = jax.random.split(key)
    if dataset.metadata.fix_x:
      dataset.build_graphs(builder=graph_builder, key=subkey)

    # Train one epoch
    subkey_0, subkey_1, key = jax.random.split(key, num=3)
    state, loss, grad = train_one_epoch(
      key=subkey_1,
      state=state,
      batches=dataset.batches(mode='train', batch_size=FLAGS.batch_size, key=subkey_0),
    )

    if ((epoch % evaluation_frequency) == 0) or (epoch == epochs):
      # Evaluate
      metrics_trn = evaluate(
        state=state,
        batches=dataset.batches(mode='train', batch_size=FLAGS.batch_size),
      )
      metrics_val = evaluate(
        state=state,
        batches=dataset.batches(mode='valid', batch_size=FLAGS.batch_size),
      )

      # Log the results
      time_tot = time() - time_int
      tdsf = FLAGS.time_downsample_factor
      lr = state.opt_state[-1].hyperparams["learning_rate"][0].item()
      logging.info('\t'.join([
        f'DRCT: {tau_max * tdsf : 02d}dt',
        f'EPCH: {epochs_before + epoch : 04d}/{FLAGS.epochs : 04d}',
        f'LR: {lr : .2e}',
        f'TIME: {time_tot : 06.1f}s',
        f'GRAD: {grad.item() : .2e}',
        f'LOSS: {loss.item() : .2e}',
        f'DR-{.5 * tdsf}: {metrics_val.direct_tau_frac._l1 : .2%}' if metrics_val.direct_tau_frac._l1 else '',
        f'DR-{tdsf}: {metrics_val.direct_tau_min._l1 : .2%}' if metrics_val.direct_tau_min._l1 else '',
        f'DR-{FLAGS.tau_max * tdsf}: {metrics_val.direct_tau_max._l1 : .2%}' if metrics_val.direct_tau_max._l1 else '',
        f'FN: {metrics_val.final._l1 : .2%}' if metrics_val.final._l1 else '',
        f'TRN-DR-{tdsf}: {metrics_trn.direct_tau_min._l1 : .2%}' if metrics_trn.direct_tau_min._l1 else '',
        f'TRN-DR-{FLAGS.tau_max * tdsf}: {metrics_trn.direct_tau_max._l1 : .2%}' if metrics_trn.direct_tau_max._l1 else '',
        f'TRN-FN: {metrics_trn.final._l1 : .2%}' if metrics_trn.final._l1 else '',
      ]))

      with disable_logging(level=logging.FATAL):
        step = epochs_before + epoch
        checkpoint_metrics = {
          'step': step,
          'loss': loss.item(),
          'lr': lr,
          'train': metrics_trn.to_dict(),
          'valid': metrics_val.to_dict(),
        }
        checkpointed_metrics.append(checkpoint_metrics)
        # Store the state and the metrics
        checkpoint_manager.save(
          step=step,
          items={'state': jax.device_get(unreplicate(state)),},
          metrics=checkpoint_metrics,
          save_kwargs={'save_args': checkpointer_save_args}
        )
        with open(DIR / 'metrics' / f'{str(step)}.json', 'w') as f:
          json.dump(checkpoint_metrics, f)
        # Plot the history of the metrics
        metrics_to_plot = {
          'optimization': (
            {'label': 'Training loss', 'values': lambda m: m['loss']},
            {'label': 'Training loss', 'values': lambda m: m['loss']},
          ),
          'final': (
            {'label': 'Training error [%]', 'values': lambda m: m['train']['final']['_l1'] * 100},
            {'label': 'Validation error [%]', 'values': lambda m: m['valid']['final']['_l1'] * 100}
          ),
          'direct': (
            {'label': 'Training error [%]', 'values': lambda m: m['train']['direct_tau_min']['_l1'] * 100 if dataset.time_dependent else 1},
            {'label': 'Validation error [%]', 'values': lambda m: m['valid']['direct_tau_min']['_l1'] * 100 if dataset.time_dependent else 1}
          ),
        }
        steps = [m['step'] for m in checkpointed_metrics]
        for filename, mtp in metrics_to_plot.items():
          fig, axs = plt.subplots(
            ncols=2,
            figsize=(10, 3),
            sharex=True,
            sharey=(filename != 'optimization'),
            tight_layout=True,
          )
          for i, item in enumerate(mtp):
            values = [item['values'](m) for m in checkpointed_metrics]
            ax: plt.Axes = axs[i]
            ax.scatter(steps, values, s=10, color='black', zorder=3)
            ax.set(
              ylabel=item['label'],
              yscale='log',
            )
            ax.grid(which='both')
          file = DIR / 'metrics/plots' / f'{filename}.pdf'
          fig.savefig(file, dpi=500, bbox_inches='tight')
          plt.close(fig)

    else:
      # Log the results
      time_tot = time() - time_int
      tdsf = FLAGS.time_downsample_factor
      logging.info('\t'.join([
        f'DRCT: {tau_max * tdsf : 02d}dt',
        f'EPCH: {epochs_before + epoch : 04d}/{FLAGS.epochs : 04d}',
        f'LR: {state.opt_state[-1].hyperparams["learning_rate"][0].item() : .2e}',
        f'TIME: {time_tot : 06.1f}s',
        f'GRAD: {grad.item() : .2e}',
        f'LOSS: {loss.item() : .2e}',
      ]))

  return unreplicate(state)

def get_model(model_configs: Mapping[str, Any], dataset: Dataset) -> AbstractOperator:
  """Build the model based on the given configurations."""

  # Set model kwargs
  if not model_configs:
    model_configs = dict(
      num_outputs=dataset.shape[-1],
      processor_steps=FLAGS.processor_steps,
      node_latent_size=FLAGS.node_latent_size,
      edge_latent_size=FLAGS.edge_latent_size,
      mlp_hidden_layers=FLAGS.mlp_hidden_layers,
      concatenate_tau=(True if dataset.time_dependent else False),
      concatenate_t=(True if dataset.time_dependent else False),
      conditioned_normalization=(True if dataset.time_dependent else False),
      cond_norm_hidden_size=16,
      p_edge_masking=FLAGS.p_edge_masking,
    )

  model = RIGNO(**model_configs)

  return model

def main(argv):
  # Check the number of arguments
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Check the available devices
  with disable_logging():
    process_index = jax.process_index()
    process_count = jax.process_count()
    local_devices = jax.local_devices()
  logging.info('JAX host: %d / %d', process_index, process_count)
  logging.info('JAX local devices: %r', local_devices)
  # We only support single-host training.
  assert process_count == 1

  # Check the inputs
  if not FLAGS.datetime:
    FLAGS.datetime = datetime.now().strftime('%Y%m%d-%H%M%S')

  # Initialize the random key
  key = jax.random.PRNGKey(FLAGS.seed)

  # Read the dataset
  subkey, key = jax.random.split(key)
  dataset = Dataset(
    datadir=FLAGS.datadir,
    datapath=FLAGS.datapath,
    include_passive_variables=False,
    concatenate_coeffs=False,
    time_cutoff_idx=(IDX_FN + 1),
    time_downsample_factor=FLAGS.time_downsample_factor,
    space_downsample_factor=FLAGS.space_downsample_factor,
    n_train=FLAGS.n_train,
    n_valid=FLAGS.n_valid,
    n_test=FLAGS.n_test,
    preload=True,
    key=subkey,
  )
  if dataset.time_dependent:
    dataset.compute_stats(residual_steps=FLAGS.tau_max)
  else:
    assert FLAGS.stepper == 'out'
    assert FLAGS.tau_max == 0
    assert FLAGS.fractional is False
    dataset.compute_stats()

  # Read the checkpoint
  if FLAGS.params:
    DIR_OLD_EXPERIMENT = DIR_EXPERIMENTS / FLAGS.params
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    step = orbax.checkpoint.CheckpointManager(DIR_OLD_EXPERIMENT / 'checkpoints', orbax_checkpointer).latest_step()
    ckpt = orbax_checkpointer.restore(directory=(DIR_OLD_EXPERIMENT / 'checkpoints' / str(step) / 'default'))
    state = ckpt['state']
    params = state['params']
    with open(DIR_OLD_EXPERIMENT / 'configs.json', 'rb') as f:
      old_configs = json.load(f)
      model_configs = old_configs['model_configs']
  else:
    params = None
    model_configs = None

  # Get the model
  model = get_model(model_configs, dataset)

  # Store the configurations
  DIR = DIR_EXPERIMENTS / f'E{FLAGS.exp}' / FLAGS.datapath / FLAGS.datetime
  DIR.mkdir(parents=True)
  logging.info(f'Experiment stored in {DIR.relative_to(DIR_EXPERIMENTS).as_posix()}')
  flags = {f: FLAGS.get_flag_value(f, default=None) for f in FLAGS}
  with open(DIR / 'configs.json', 'w') as f:
    json.dump(fp=f,
      obj={'flags': flags, 'model_configs': model.configs},
      indent=2,
    )
  # Store the statistics
  with open(DIR / 'stats.pkl', 'wb') as f:
    pickle.dump(file=f, obj=dataset.stats)

  # Construct the graphs
  logging.info('Constructing the graphs for all dataset samples...')
  graph_builder = RegionInteractionGraphBuilder(
    periodic=dataset.metadata.periodic,
    rmesh_levels=FLAGS.rmesh_levels,
    subsample_factor=FLAGS.mesh_subsample_factor,
    overlap_factor_p2r=FLAGS.overlap_factor_p2r,
    overlap_factor_r2p=FLAGS.overlap_factor_r2p,
    node_coordinate_freqs=FLAGS.node_coordinate_freqs,
  )
  dataset.build_graphs(builder=graph_builder)
  logging.info(f'Constructed {len(dataset.rigs)} graph(s).')

  schedule_tau_max = True
  if (FLAGS.tau_max == 1):
    schedule_tau_max = False
  if not dataset.time_dependent:
    schedule_tau_max = False

  # Split the epochs
  if schedule_tau_max:
    epochs_warmup = int(.2 * FLAGS.epochs)
    epochs_dxx = epochs_warmup // (FLAGS.tau_max - 1)
    epochs_dff = (FLAGS.epochs - epochs_warmup) + epochs_warmup % (FLAGS.tau_max - 1)

  # Initialzize the model or use the loaded parameters
  if not params:
    dummy_graph_builder = RegionInteractionGraphBuilder(
      periodic=dataset.metadata.periodic,
      rmesh_levels=1,
      subsample_factor=4,
      overlap_factor_p2r=.01,
      overlap_factor_r2p=.01,
      node_coordinate_freqs=FLAGS.node_coordinate_freqs,
    )
    dummy_graphs = dummy_graph_builder.build_graphs(
        dummy_graph_builder.build_metadata(
        x_inp=dataset.sample.x[0, 0],
        x_out=dataset.sample.x[0, 0],
        domain=np.array(dataset.metadata.domain_x),
      )
    )
    dummy_graphs = tree.tree_map(lambda v: jnp.repeat(v, repeats=FLAGS.batch_size, axis=0), dummy_graphs)
    if dataset.time_dependent:
      dummy_inputs = Inputs(
        u=jnp.ones(shape=(FLAGS.batch_size, 1, *dataset.sample.u.shape[2:])),
        c=jnp.ones(shape=(FLAGS.batch_size, 1, *dataset.sample.c.shape[2:])) if (dataset.sample.c is not None) else None,
        x_inp=dataset.sample.x,
        x_out=dataset.sample.x,
        t=jnp.zeros(shape=(FLAGS.batch_size, 1)),
        tau=jnp.ones(shape=(FLAGS.batch_size, 1)),
      )
    else:
      dummy_inputs = Inputs(
        u=jnp.ones(shape=(FLAGS.batch_size, 1, *dataset.sample.c.shape[2:])),
        c=None,
        x_inp=dataset.sample.x,
        x_out=dataset.sample.x,
        t=None,
        tau=None,
      )
    subkey, key = jax.random.split(key)
    variables = jax.jit(model.init)(subkey, inputs=dummy_inputs, graphs=dummy_graphs)
    params = variables['params']

  # Calculate the total number of parameters
  n_model_parameters = np.sum(
    jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: np.prod(x.shape).item(), params))[0]
  ).item()
  logging.info(f'Training a {model.__class__.__name__} with {n_model_parameters} parameters')

  # Set transition steps
  num_batches = dataset.nums['train'] // FLAGS.batch_size
  if dataset.time_dependent:
    num_times = dataset.shape[1]
    num_init_times = num_times - 1
    assert num_init_times > 0
    num_init_times_full = max(0, num_times - FLAGS.tau_max)
    num_init_times_part = num_init_times - num_init_times_full
    transition_steps = 0
    for _d in (range(1, FLAGS.tau_max+1) if schedule_tau_max else [FLAGS.tau_max]):
      num_valid_pairs_d = (
        num_init_times_full * _d
        + (num_init_times_part * (num_init_times_part+1) // 2)
      )
      if schedule_tau_max:
        epochs_d = (epochs_dff if (_d == FLAGS.tau_max) else epochs_dxx)
      else:
        epochs_d = FLAGS.epochs
      transition_steps +=  epochs_d * num_batches * num_valid_pairs_d
  else:
    transition_steps = FLAGS.epochs * num_batches

  # Set learning rate and optimizer
  pct_start = .02  # Warmup cosine onecycle
  pct_final = .1   # Final exponential decay
  lr = optax.join_schedules(
    schedules=[
      optax.cosine_onecycle_schedule(
        transition_steps=((1 - pct_final) * transition_steps),
        peak_value=FLAGS.lr_peak,
        pct_start=(pct_start / (1 - pct_final)),
        div_factor=(FLAGS.lr_peak / FLAGS.lr_init),
        final_div_factor=(FLAGS.lr_init / FLAGS.lr_base),
      ),
      optax.exponential_decay(
        transition_steps=(pct_final * transition_steps),
        init_value=FLAGS.lr_base,
        decay_rate=(FLAGS.lr_lowr / FLAGS.lr_base) if FLAGS.lr_lowr else 1,
      ),
    ],
    boundaries=[int((1 - pct_final) * transition_steps),],
  )
  tx = optax.chain(
    optax.inject_hyperparams(optax.adamw)(learning_rate=lr, weight_decay=1e-08),
  )
  state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

  # Train the model
  epochs_trained = 0

  # Warm-up epochs
  if schedule_tau_max:
    for _d in range(1, FLAGS.tau_max):
      key, subkey = jax.random.split(key)
      state = train(
        key=subkey,
        model=model,
        state=state,
        dataset=dataset,
        graph_builder=graph_builder,
        tau_max=_d,
        unroll=False,
        epochs=epochs_dxx,
        epochs_before=epochs_trained,
      )
      epochs_trained += epochs_dxx

  # Split with and without unrolling
  if schedule_tau_max:
    epochs_rest = epochs_dff
  else:
    epochs_rest = FLAGS.epochs
  epochs_with_unrolling = int(.5 * epochs_rest)
  epochs_without_unrolling = epochs_rest - epochs_with_unrolling

  if FLAGS.fractional:
    # Train without unrolling
    state = train(
      key=subkey,
      model=model,
      state=state,
      dataset=dataset,
      graph_builder=graph_builder,
      tau_max=FLAGS.tau_max,
      unroll=False,
      epochs=epochs_without_unrolling,
      epochs_before=epochs_trained,
    )
    epochs_trained += epochs_without_unrolling
    # Train with unrolling
    logging.info('-' * 80)
    logging.info('FRACTIONAL PAIRING')
    logging.info('-' * 80)
    state = train(
      key=subkey,
      model=model,
      state=state,
      dataset=dataset,
      graph_builder=graph_builder,
      tau_max=FLAGS.tau_max,
      unroll=True,
      epochs=epochs_with_unrolling,
      epochs_before=epochs_trained,
    )
    epochs_trained += epochs_with_unrolling
  else:
    # Train without unrolling
    state = train(
      key=subkey,
      model=model,
      state=state,
      dataset=dataset,
      graph_builder=graph_builder,
      tau_max=FLAGS.tau_max,
      unroll=False,
      epochs=epochs_rest,
      epochs_before=epochs_trained,
    )
    epochs_trained += epochs_rest

if __name__ == '__main__':
  logging.set_verbosity('info')
  define_flags()
  app.run(main)
