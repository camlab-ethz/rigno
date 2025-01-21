import json
import pickle
import shutil
from time import time
from typing import Type, Mapping, Callable, Any, Sequence

import jax
import jax.numpy as jnp
import jax.tree_util as tree
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint
import pandas as pd
from absl import app, flags, logging
from flax.training.common_utils import shard, shard_prng_key
from flax.typing import PRNGKey
from flax.jax_utils import replicate

from rigno.dataset import Dataset, Batch
from rigno.experiments import DIR_EXPERIMENTS
from rigno.metrics import rel_lp_error
from rigno.models.operator import AbstractOperator, Inputs
from rigno.models.rigno import RIGNO, RegionInteractionGraphMetadata, RegionInteractionGraphBuilder
from rigno.plot import plot_ensemble, plot_error_vs_time, plot_estimates
from rigno.stepping import Stepper, TimeDerivativeStepper, ResidualStepper, OutputStepper
from rigno.stepping import AutoregressiveStepper
from rigno.utils import Array, disable_logging, profile


NUM_DEVICES = jax.local_device_count()
IDX_FN = 14

FLAGS = flags.FLAGS

def define_flags():
  flags.DEFINE_string(name='exp', default=None, required=True,
    help='Relative path of the experiment'
  )
  flags.DEFINE_string(name='datadir', default=None, required=True,
    help='Path of the folder containing the datasets'
  )
  flags.DEFINE_integer(name='batch_size', default=4, required=False,
    help='Size of a batch of training samples'
  )
  flags.DEFINE_integer(name='n_test', default=(2**8), required=False,
    help='Number of test samples'
  )
  flags.DEFINE_boolean(name='only_profile', default=False, required=False,
    help='If passed, the tests are skipped'
  )
  flags.DEFINE_boolean(name='resolution', default=False, required=False,
    help='If passed, estimations with different discretizations are computed'
  )
  flags.DEFINE_boolean(name='noise', default=False, required=False,
    help='If passed, estimations for noise control are computed'
  )
  flags.DEFINE_boolean(name='ensemble', default=False, required=False,
    help='If passed, ensemble samples are generated using model randomness'
  )

def _print_between_dashes(msg):
  logging.info('-' * 80)
  logging.info(msg)
  logging.info('-' * 80)

def _build_graph_metadata(batch: Batch, graph_builder: RegionInteractionGraphBuilder, dataset: Dataset, rmesh_correction_dsf: int = 1) -> RegionInteractionGraphMetadata:
  # Build graph metadata with transformed coordinates
  metadata = []
  num_p2r_edges = 0
  num_r2r_edges = 0
  num_r2p_edges = 0
  # Loop over all coordinates in the batch
  # NOTE: Assuming constant x in time
  for x in batch.x[:, 0]:
    m = graph_builder.build_metadata(x_inp=x, x_out=x, domain=np.array(dataset.metadata.domain_x), rmesh_correction_dsf=rmesh_correction_dsf)
    metadata.append(m)
    # Store the maximum number of edges
    num_p2r_edges = max(num_p2r_edges, m.p2r_edge_indices.shape[1])
    num_r2r_edges = max(num_r2r_edges, m.r2r_edge_indices.shape[1])
    if m.r2p_edge_indices is not None:
      num_r2p_edges = max(num_r2p_edges, m.r2p_edge_indices.shape[1])
  # Pad the edge sets using dummy nodes
  # NOTE: Exploiting jax' behavior for out-of-dimension indexing
  for i, m in enumerate(metadata):
    m: RegionInteractionGraphMetadata
    metadata[i] = RegionInteractionGraphMetadata(
      x_pnodes_inp=m.x_pnodes_inp,
      x_pnodes_out=m.x_pnodes_out,
      x_rnodes=m.x_rnodes,
      r_rnodes=m.r_rnodes,
      p2r_edge_indices=m.p2r_edge_indices[:, jnp.arange(num_p2r_edges), :],
      r2r_edge_indices=m.r2r_edge_indices[:, jnp.arange(num_r2r_edges), :],
      r2r_edge_domains=m.r2r_edge_domains[:, jnp.arange(num_r2r_edges), :],
      r2p_edge_indices=m.r2p_edge_indices[:, jnp.arange(num_r2p_edges), :] if (m.r2p_edge_indices is not None) else None,
    )
  # Concatenate all padded graph sets and store them
  g = tree.tree_map(lambda *v: jnp.concatenate(v), *metadata)

  return g

def _change_discretization(batch: Batch, key: PRNGKey = None):
  if key is None:
    key = jax.random.PRNGKey(0)
  permutation = jax.random.permutation(key, batch.shape[2])
  _u = batch.u[:, :, permutation, :]
  _c = batch.c[:, :, permutation, :] if (batch.c is not None) else None
  _x = batch.x[:, :, permutation, :]
  _g = None
  return Batch(u=_u, c=_c, x=_x, t=batch.t, g=_g)

def _change_resolution(batch: Batch, space_downsample_factor: int):
  if space_downsample_factor == 1:
    return batch
  num_space = int(batch.shape[2] / space_downsample_factor)
  batch = _change_discretization(batch)
  _u = batch.u[:, :, :num_space, :]
  _c = batch.c[:, :, :num_space, :] if (batch.c is not None) else None
  _x = batch.x[:, :, :num_space, :]
  _g = None
  return Batch(u=_u, c=_c, x=_x, t=batch.t, g=_g)

def profile_inference(
  dataset: Dataset,
  graph_builder: RegionInteractionGraphBuilder,
  model: AbstractOperator,
  stepping: Type[Stepper],
  state: dict,
  stats: dict,
  p_edge_masking: float,
  repeats: int = 10,
  jit: bool = True,
):

  # Configure and build new model
  model_configs = model.configs
  model_configs['p_edge_masking'] = p_edge_masking
  stepper = stepping(operator=RIGNO(**model_configs))

  apply_fn = stepper.apply
  if jit: apply_fn = jax.jit(apply_fn)
  graph_fn = lambda x: graph_builder.build_metadata(x, x, np.array(dataset.metadata.domain_x))

  # Get a batch and transform it
  batch_size_per_device = 1
  batch = next(dataset.batches(mode='test', batch_size=batch_size_per_device))

  # Set model inputs
  if dataset.time_dependent:
    model_kwargs = dict(
      variables={'params': state['params']},
      stats=stats,
      inputs=Inputs(
        u=batch.u[:, [0]],
        c=(batch.c[:, [0]] if (batch.c is not None) else None),
        x_inp=batch.x,
        x_out=batch.x,
        t=batch.t[:, [0]],
        tau=dataset.dt,
      ),
      graphs=graph_builder.build_graphs(batch.g),
      key=jax.random.PRNGKey(0),
    )
  else:
    model_kwargs = dict(
      variables={'params': state['params']},
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
      key=jax.random.PRNGKey(0),
    )

  # Profile graph building
  t_graph = profile(graph_fn, kwargs=dict(x=batch.x[0, 0]), repeats=10)
  # Profile compilation
  t_compilation = profile(f=apply_fn, kwargs=model_kwargs, repeats=1, block_until_ready=True)
  # Profile inferrence after compilation
  t = profile(f=apply_fn, kwargs=model_kwargs, repeats=repeats, block_until_ready=True)

  general_info = [
    'NUMBER OF DEVICES: 1',
    f'BATCH SIZE PER DEVICE: {batch_size_per_device}',
    f'MODEL: {model.__class__.__name__}',
    f'p_edge_masking: {p_edge_masking}',
  ]

  times_info = [
    f'Graph building: {t_graph * 1000: .2f}ms',
    f'Compilation: {t_compilation : .2f}s',
    f'Inferrence: {t * 1000 : .2f}ms per batch',
    f'Inferrence: {t * 1000 / batch_size_per_device : .2f}ms per sample',
  ]

  # Print all messages in dashes
  def wrap_in_dashes(lines, width):
    return ['-' * width] + lines + ['-' * width]
  all_msgs = wrap_in_dashes(general_info, 80) + wrap_in_dashes(times_info, 80)
  for line in all_msgs:
    logging.info(line)

def get_time_independent_estimations(
  step: Stepper.apply,
  graph_builder: RegionInteractionGraphBuilder,
  variables,
  stats,
  batch: Batch,
  key = None,
) -> Array:

  inputs = Inputs(
    u=batch.c[:, [0]],
    c=None,
    x_inp=batch.x,
    x_out=batch.x,
    t=None,
    tau=None,
  )

  _u_prd = step(
    variables=variables,
    stats=stats,
    inputs=inputs,
    graphs=graph_builder.build_graphs(batch.g),
    key=key,
  )

  return _u_prd

def get_direct_estimations(
  step: Stepper.apply,
  graph_builder: RegionInteractionGraphBuilder,
  variables,
  stats,
  batch: Batch,
  tau: float,
  key = None,
) -> Array:
  """Inputs are of shape [batch_size_per_device, ...]"""

  # Set lead times
  lead_times = jnp.arange(batch.shape[1])
  batch_size = batch.shape[0]

  # Get inputs for all lead times
  # -> [num_lead_times, batch_size_per_device, ...]
  u_inp = jax.vmap(
      lambda lt: jax.lax.dynamic_slice_in_dim(
        operand=batch.u,
        start_index=(lt), slice_size=1, axis=1)
  )(lead_times)
  t_inp = batch.t.swapaxes(0, 1).reshape(-1, batch_size, 1)

  # Get model estimations
  def _use_step_on_mini_batches(carry, x):
    idx = carry
    inputs = Inputs(
      u=u_inp[idx],
      c=(batch.c[:, [idx]] if (batch.c is not None) else None),
      x_inp=batch.x,
      x_out=batch.x,
      t=t_inp[idx],
      tau=tau,
    )
    _u_prd = step(
      variables=variables,
      stats=stats,
      inputs=inputs,
      graphs=graph_builder.build_graphs(batch.g),
      key=key,
    )
    carry += 1
    return carry, _u_prd
  # -> [num_lead_times, batch_size_per_device, 1, ...]
  _, u_prd = jax.lax.scan(
    f=_use_step_on_mini_batches,
    init=0,
    xs=None,
    length=batch.shape[1],
  )

  # Re-arrange
  # -> [batch_size_per_device, num_lead_times, ...]
  u_prd = u_prd.swapaxes(0, 1).squeeze(axis=2)

  return u_prd

def get_rollout_estimations(
  unroll: AutoregressiveStepper.unroll,
  num_steps: int,
  graph_builder: RegionInteractionGraphBuilder,
  variables,
  stats,
  batch: Batch,
  key = None,
):
  """Inputs are of shape [batch_size_per_device, ...]"""

  inputs = Inputs(
    u=batch.u[:, [0]],
    c=(batch.c[:, [0]] if (batch.c is not None) else None),
    x_inp=batch.x,
    x_out=batch.x,
    t=batch.t[:, [0]],
    tau=None,
  )
  rollout, _ = unroll(
    variables,
    stats,
    num_steps,
    inputs=inputs,
    key=key,
    graphs=graph_builder.build_graphs(batch.g),
  )

  return rollout

def get_all_estimations(
  dataset: Dataset,
  model: AbstractOperator,
  graph_builder: RegionInteractionGraphBuilder,
  stepping: Type[Stepper],
  state: dict,
  stats: dict,
  train_flags: Mapping,
  taus_direct: Sequence[int] = [],
  taus_rollout: Sequence[int] = [],
  space_dsfs: Sequence[int] = [],
  noise_levels: Sequence[float] = [],
  p_edge_masking: float = 0.,
):

  # Replicate state and stats
  # NOTE: Internally uses jax.device_put_replicate
  state = replicate(state)
  stats = replicate(stats)

  # Get pmapped version of the estimator functions
  _get_tindep_estimations = jax.pmap(get_time_independent_estimations, static_broadcasted_argnums=(0, 1))
  _get_direct_estimations = jax.pmap(get_direct_estimations, static_broadcasted_argnums=(0, 1))
  _get_rollout_estimations = jax.pmap(get_rollout_estimations, static_broadcasted_argnums=(0, 1, 2))

  def _get_estimations_in_batches(
    direct: bool,
    apply_fn: Callable,
    tau_ratio: int = None,
    transform: Callable[[Array], Array] = None,
    dsf: int = 1,
  ):
    # Check inputs
    if dataset.time_dependent and direct:
      assert tau_ratio is not None

    # Loop over the batches
    u_prd = []
    for batch in dataset.batches(mode='test', batch_size=FLAGS.batch_size):
      batch: Batch
      # Transform the batch
      if transform is not None:
        batch = transform(batch)
        correction_dsf = train_flags['space_downsample_factor'] / dsf
        g = _build_graph_metadata(batch, graph_builder, dataset, rmesh_correction_dsf=correction_dsf)
      else:
        g = batch.g

      # Split the batch between devices
      # -> [NUM_DEVICES, batch_size_per_device, ...]
      batch = Batch(
        u=shard(batch.u),
        c=shard(batch.c),
        x=shard(batch.x),
        t=shard(batch.t),
        g=shard(g),
      )

      if dataset.time_dependent:
        # Get the direct predictions
        if direct:
          u_prd_batch = _get_direct_estimations(
            apply_fn,
            graph_builder,
            variables={'params': state['params']},
            stats=stats,
            batch=batch,
            tau=replicate(tau_ratio * dataset.dt),
          )
          # Replace the tail of the predictions with the head of the input
          u_prd_batch = jnp.concatenate([batch.u[:, :, :tau_ratio], u_prd_batch[:, :, :-tau_ratio]], axis=2)

        # Get the rollout predictions
        else:
          num_times = batch.shape[2]
          u_prd_batch = _get_rollout_estimations(
            apply_fn,
            num_times,
            graph_builder,
            variables={'params': state['params']},
            stats=stats,
            batch=batch,
          )

      else:
        u_prd_batch = _get_tindep_estimations(
          apply_fn,
          graph_builder,
          variables={'params': state['params']},
          stats=stats,
          batch=batch,
        )

      # Undo the split between devices
      u_prd_batch = u_prd_batch.reshape(FLAGS.batch_size, *u_prd_batch.shape[2:])

      # Append the prediction
      u_prd.append(u_prd_batch)

    # Concatenate the predictions
    u_prd = jnp.concatenate(u_prd, axis=0)

    return u_prd

  # Instantiate the steppers
  all_dsfs = set(space_dsfs + [train_flags['space_downsample_factor']])
  steppers: dict[Any, Stepper] = {res: None for res in all_dsfs}
  apply_steppers_jit: dict[Any, Stepper.apply] = {res: None for res in all_dsfs}
  apply_steppers_twice_jit: dict[Any, Stepper.unroll] = {res: None for res in all_dsfs}
  unrollers: dict[Any, dict[Any, AutoregressiveStepper]] = {
    res: {tau: None for tau in taus_rollout} for res in all_dsfs}
  apply_unroll_jit: dict[Any, dict[Any, AutoregressiveStepper.unroll]] = {
    res: {tau: None for tau in taus_rollout} for res in all_dsfs}

  # Instantiate the steppers
  # TODO: Same stepper for all resolutions !!
  for dsf in all_dsfs:
    # Configure and build new model
    model_configs = model.configs
    model_configs['p_edge_masking'] = p_edge_masking

    steppers[dsf] = stepping(operator=model.__class__(**model_configs))
    apply_steppers_jit[dsf] = jax.jit(steppers[dsf].apply)
    def apply_steppers_twice(*args, **kwargs):
      return steppers[dsf].unroll(*args, **kwargs, num_steps=2)
    apply_steppers_twice_jit[dsf] = jax.jit(apply_steppers_twice)

    for tau_ratio_max in taus_rollout:
      unrollers[dsf][tau_ratio_max] = AutoregressiveStepper(
        stepper=steppers[dsf],
        dt=dataset.dt,
        tau_max=(tau_ratio_max * dataset.dt),
      )
      apply_unroll_jit[dsf][tau_ratio_max] = jax.jit(
        unrollers[dsf][tau_ratio_max].unroll, static_argnums=(2,))

  # Set the ground-truth solutions
  batch_test = next(dataset.batches(mode='test', batch_size=dataset.nums['test'], get_graphs=False))

  # Instantiate the outputs
  errors = {error_type: {
      key: {'direct': {}, 'rollout': {}} if dataset.time_dependent else {}
      for key in ['tau', 'disc', 'dsf', 'noise']
    }
    for error_type in ['_l1', '_l2']
  }
  u_prd_output = None

  # Get predictions for plotting
  tau_ratio_max = train_flags['time_downsample_factor']
  dsf = train_flags['space_downsample_factor']
  if dataset.time_dependent:
    u_prd_output = _get_estimations_in_batches(
      direct=False,
      apply_fn=apply_unroll_jit[dsf][tau_ratio_max],
      transform=(lambda b: _change_resolution(b, dsf)),
      dsf=dsf,
    )[:, [0, IDX_FN, -1]]
  else:
    u_prd_output = _get_estimations_in_batches(
      direct=False,
      apply_fn=apply_steppers_jit[dsf],
      transform=(lambda b: _change_resolution(b, dsf)),
      dsf=dsf,
    )

  # Define a auxiliary function for getting the errors
  def _get_err_trajectory(_u_gtr, _u_prd, p):
    mean = np.array(dataset.metadata.global_mean)
    std = np.array(dataset.metadata.global_std)
    _u_gtr = (_u_gtr - mean) / std
    _u_prd = (_u_prd - mean) / std
    _err = [
      np.mean(np.median(rel_lp_error(
        _u_gtr[:, [idx_t]],
        _u_prd[:, [idx_t]],
        p=p,
        chunks=dataset.metadata.chunked_variables,
        num_chunks=dataset.metadata.num_variable_chunks,
      ), axis=0)).item() * 100
      for idx_t in range(_u_gtr.shape[1])
    ]
    return _err

  # Temporal continuity (time-dependent)
  dsf = train_flags['space_downsample_factor']
  for tau_ratio in taus_direct:
    if dataset.time_dependent:
      if tau_ratio == .5:
        _apply_stepper = apply_steppers_twice_jit[dsf]
      else:
        _apply_stepper = apply_steppers_jit[dsf]
      t0 = time()
      u_prd = _get_estimations_in_batches(
        direct=True,
        apply_fn=_apply_stepper,
        tau_ratio=(tau_ratio if tau_ratio != .5 else 1),
        transform=(lambda b: _change_resolution(b, dsf)),
        dsf=dsf,
      )
      errors['_l1']['tau']['direct'][tau_ratio] = _get_err_trajectory(
        _u_gtr=_change_resolution(batch_test, space_downsample_factor=dsf).u,
        _u_prd=u_prd,
        p=1,
      )
      errors['_l2']['tau']['direct'][tau_ratio] = _get_err_trajectory(
        _u_gtr=_change_resolution(batch_test, space_downsample_factor=dsf).u,
        _u_prd=u_prd,
        p=2,
      )
      del u_prd
      _print_between_dashes(f'tau_direct={tau_ratio} \t TIME={time()-t0 : .4f}s')

  # Autoregressive rollout (time-dependent)
  dsf = train_flags['space_downsample_factor']
  for tau_ratio_max in taus_rollout:
    if dataset.time_dependent:
      t0 = time()
      u_prd = _get_estimations_in_batches(
        direct=False,
        apply_fn=apply_unroll_jit[dsf][tau_ratio_max],
        transform=(lambda b: _change_resolution(b, dsf)),
        dsf=dsf,
      )
      errors['_l1']['tau']['rollout'][tau_ratio_max] = _get_err_trajectory(
        _u_gtr=_change_resolution(batch_test, space_downsample_factor=dsf).u,
        _u_prd=u_prd,
        p=1,
      )
      errors['_l2']['tau']['rollout'][tau_ratio_max] = _get_err_trajectory(
        _u_gtr=_change_resolution(batch_test, space_downsample_factor=dsf).u,
        _u_prd=u_prd,
        p=2,
      )
      del u_prd
      _print_between_dashes(f'tau_max={tau_ratio_max} \t TIME={time()-t0 : .4f}s')

  # Discretization invariance
  tau_ratio_max = train_flags['time_downsample_factor']
  tau_ratio = train_flags['time_downsample_factor']
  dsf = train_flags['space_downsample_factor']
  for i_disc in range(4):
    key = jax.random.PRNGKey(i_disc)
    if dataset.time_dependent:
      # Direct
      t0 = time()
      u_prd = _get_estimations_in_batches(
        direct=True,
        apply_fn=apply_steppers_jit[dsf],
        tau_ratio=tau_ratio,
        transform=(lambda b: _change_resolution(_change_discretization(b, key), dsf)),
        dsf=dsf,
      )
      errors['_l1']['disc']['direct'][i_disc] = _get_err_trajectory(
        _u_gtr=_change_resolution(_change_discretization(batch_test, key), dsf).u,
        _u_prd=u_prd,
        p=1,
      )
      errors['_l2']['disc']['direct'][i_disc] = _get_err_trajectory(
        _u_gtr=_change_resolution(_change_discretization(batch_test, key), dsf).u,
        _u_prd=u_prd,
        p=2,
      )
      del u_prd
      _print_between_dashes(f'discretization={i_disc} (direct) \t TIME={time()-t0 : .4f}s')
      # Rollout
      t0 = time()
      u_prd = _get_estimations_in_batches(
        direct=False,
        apply_fn=apply_unroll_jit[dsf][tau_ratio_max],
        transform=(lambda b: _change_resolution(_change_discretization(b, key), dsf)),
        dsf=dsf,
      )
      errors['_l1']['disc']['rollout'][i_disc] = _get_err_trajectory(
        _u_gtr=_change_resolution(_change_discretization(batch_test, key), dsf).u,
        _u_prd=u_prd,
        p=1,
      )
      errors['_l2']['disc']['rollout'][i_disc] = _get_err_trajectory(
        _u_gtr=_change_resolution(_change_discretization(batch_test, key), dsf).u,
        _u_prd=u_prd,
        p=2,
      )
      del u_prd
      _print_between_dashes(f'discretization={i_disc} (rollout) \t TIME={time()-t0 : .4f}s')
    else:
      t0 = time()
      u_prd = _get_estimations_in_batches(
        direct=False,
        apply_fn=apply_steppers_jit[dsf],
        transform=(lambda b: _change_resolution(_change_discretization(b, key), dsf)),
        dsf=dsf,
      )
      errors['_l1']['disc'][i_disc] = _get_err_trajectory(
        _u_gtr=_change_resolution(_change_discretization(batch_test, key), dsf).u,
        _u_prd=u_prd,
        p=1,
      )
      errors['_l2']['disc'][i_disc] = _get_err_trajectory(
        _u_gtr=_change_resolution(_change_discretization(batch_test, key), dsf).u,
        _u_prd=u_prd,
        p=2,
      )
      del u_prd
      _print_between_dashes(f'discretization={i_disc} \t TIME={time()-t0 : .4f}s')

  # Resolution invariance
  tau_ratio_max = train_flags['time_downsample_factor']
  tau_ratio = train_flags['time_downsample_factor']
  for dsf in space_dsfs:
    if dataset.time_dependent:
      # Direct
      t0 = time()
      u_prd = _get_estimations_in_batches(
        direct=True,
        apply_fn=apply_steppers_jit[dsf],
        tau_ratio=tau_ratio,
        transform=(lambda b: _change_resolution(b, dsf)),
        dsf=dsf,
      )
      errors['_l1']['dsf']['direct'][dsf] = _get_err_trajectory(
        _u_gtr=_change_resolution(batch_test, space_downsample_factor=dsf).u,
        _u_prd=u_prd,
        p=1,
      )
      errors['_l2']['dsf']['direct'][dsf] = _get_err_trajectory(
        _u_gtr=_change_resolution(batch_test, space_downsample_factor=dsf).u,
        _u_prd=u_prd,
        p=2,
      )
      del u_prd
      _print_between_dashes(f'dsf={dsf} (direct) \t TIME={time()-t0 : .4f}s')
      # Rollout
      t0 = time()
      u_prd = _get_estimations_in_batches(
        direct=False,
        apply_fn=apply_unroll_jit[dsf][tau_ratio_max],
        transform=(lambda b: _change_resolution(b, dsf)),
        dsf=dsf,
      )
      errors['_l1']['dsf']['rollout'][dsf] = _get_err_trajectory(
        _u_gtr=_change_resolution(batch_test, space_downsample_factor=dsf).u,
        _u_prd=u_prd,
        p=1,
      )
      errors['_l2']['dsf']['rollout'][dsf] = _get_err_trajectory(
        _u_gtr=_change_resolution(batch_test, space_downsample_factor=dsf).u,
        _u_prd=u_prd,
        p=2,
      )
      del u_prd
      _print_between_dashes(f'dsf={dsf} (rollout) \t TIME={time()-t0 : .4f}s')
    else:
      t0 = time()
      u_prd = _get_estimations_in_batches(
        direct=False,
        apply_fn=apply_steppers_jit[dsf],
        transform=(lambda b: _change_resolution(b, dsf)),
        dsf=dsf,
      )
      errors['_l1']['dsf'][dsf] = _get_err_trajectory(
        _u_gtr=_change_resolution(batch_test, space_downsample_factor=dsf).u,
        _u_prd=u_prd,
        p=1,
      )
      errors['_l2']['dsf'][dsf] = _get_err_trajectory(
        _u_gtr=_change_resolution(batch_test, space_downsample_factor=dsf).u,
        _u_prd=u_prd,
        p=2,
      )
      del u_prd
      _print_between_dashes(f'dsf={dsf} \t TIME={time()-t0 : .4f}s')

  # Robustness to noise
  tau_ratio_max = train_flags['time_downsample_factor']
  tau_ratio = train_flags['time_downsample_factor']
  dsf = train_flags['space_downsample_factor']
  for noise_level in noise_levels:
    # Transformation
    def transform(batch):
      batch = _change_resolution(batch, dsf)
      u_std = np.std(batch.u, axis=(0, 2), keepdims=True)
      u_noisy = batch.u + noise_level * np.random.normal(scale=u_std, size=batch.shape)
      if batch.c is not None:
        c_std = np.std(batch.c, axis=(0, 2), keepdims=True)
        c_noisy = batch.c + noise_level * np.random.normal(scale=c_std, size=batch.shape)
      batch_noisy = Batch(
        u=u_noisy,
        c=(c_noisy if (batch.c is not None) else None),
        x=batch.x,
        t=batch.t,
        g=batch.g,
      )
      return batch_noisy
    if dataset.time_dependent:
      # Direct estimations
      t0 = time()
      u_prd = _get_estimations_in_batches(
        direct=True,
        apply_fn=apply_steppers_jit[dsf],
        tau_ratio=tau_ratio,
        transform=transform,
        dsf=dsf,
      )
      errors['_l1']['noise']['direct'][noise_level] = _get_err_trajectory(
        _u_gtr=_change_resolution(batch_test, space_downsample_factor=dsf).u,
        _u_prd=u_prd,
        p=1,
      )
      errors['_l2']['noise']['direct'][noise_level] = _get_err_trajectory(
        _u_gtr=_change_resolution(batch_test, space_downsample_factor=dsf).u,
        _u_prd=u_prd,
        p=2,
      )
      del u_prd
      _print_between_dashes(f'noise_level={noise_level} (direct) \t TIME={time()-t0 : .4f}s')
      # Rollout estimations
      t0 = time()
      u_prd = _get_estimations_in_batches(
        direct=False,
        apply_fn=apply_unroll_jit[dsf][tau_ratio_max],
        transform=transform,
        dsf=dsf,
      )
      errors['_l1']['noise']['rollout'][noise_level] = _get_err_trajectory(
        _u_gtr=_change_resolution(batch_test, space_downsample_factor=dsf).u,
        _u_prd=u_prd,
        p=1,
      )
      errors['_l2']['noise']['rollout'][noise_level] = _get_err_trajectory(
        _u_gtr=_change_resolution(batch_test, space_downsample_factor=dsf).u,
        _u_prd=u_prd,
        p=2,
      )
      del u_prd
      _print_between_dashes(f'noise_level={noise_level} (rollout) \t TIME={time()-t0 : .4f}s')
    else:
      t0 = time()
      u_prd = _get_estimations_in_batches(
        direct=False,
        apply_fn=apply_steppers_jit[dsf],
        transform=transform,
        dsf=dsf,
      )
      errors['_l1']['noise'][noise_level] = _get_err_trajectory(
        _u_gtr=_change_resolution(batch_test, space_downsample_factor=dsf).u,
        _u_prd=u_prd,
        p=1,
      )
      errors['_l2']['noise'][noise_level] = _get_err_trajectory(
        _u_gtr=_change_resolution(batch_test, space_downsample_factor=dsf).u,
        _u_prd=u_prd,
        p=2,
      )
      del u_prd
      _print_between_dashes(f'noise_level={noise_level} \t TIME={time()-t0 : .4f}s')

  return errors, u_prd_output

def get_ensemble_estimations(
  repeats: int,
  dataset: Dataset,
  model: AbstractOperator,
  graph_builder: RegionInteractionGraphBuilder,
  stepping: Type[Stepper],
  state: dict,
  stats: dict,
  train_flags: Mapping,
  tau_ratio_max: int,
  p_edge_masking: float,
  key,
):

  # Replicate state and stats
  # NOTE: Internally uses jax.device_put_replicate
  state = replicate(state)
  stats = replicate(stats)

  # Get pmapped version of the estimator functions
  _get_tindep_estimations = jax.pmap(get_time_independent_estimations, static_broadcasted_argnums=(0, 1))
  _get_rollout_estimations = jax.pmap(get_rollout_estimations, static_broadcasted_argnums=(0, 1, 2))

  def _get_estimations_in_batches(
    apply_fn: Callable,
    transform: Callable[[Array], Array] = None,
    key = None,
  ):
    # Loop over the batches
    u_prd = []
    for batch in dataset.batches(mode='test', batch_size=FLAGS.batch_size):
      # Transform the batch
      if transform is not None:
        batch = transform(batch)
        g = _build_graph_metadata(batch, graph_builder, dataset)
      else:
        g = batch.g

      # Split the batch between devices
      # -> [NUM_DEVICES, batch_size_per_device, ...]
      batch = Batch(
        u=shard(batch.u),
        c=shard(batch.c),
        x=shard(batch.x),
        t=shard(batch.t),
        g=shard(g),
      )
      subkey, key = jax.random.split(key)
      subkey = shard_prng_key(subkey)

      # Get the rollout predictions
      if dataset.time_dependent:
        num_times = batch.shape[2]
        u_prd_batch = _get_rollout_estimations(
          apply_fn,
          num_times,
          graph_builder,
          variables={'params': state['params']},
          stats=stats,
          batch=batch,
          key=subkey,
        )
      else:
        u_prd_batch = _get_tindep_estimations(
          apply_fn,
          graph_builder,
          variables={'params': state['params']},
          stats=stats,
          batch=batch,
          key=subkey
        )

      # Undo the split between devices
      u_prd_batch = u_prd_batch.reshape(FLAGS.batch_size, *u_prd_batch.shape[2:])

      # Append the prediction
      u_prd.append(u_prd_batch)

    # Concatenate the predictions
    u_prd = jnp.concatenate(u_prd, axis=0)

    return u_prd

  # Configure and build new model
  model_configs = model.configs
  model_configs['p_edge_masking'] = p_edge_masking

  stepper = stepping(operator=model.__class__(**model_configs))
  if dataset.time_dependent:
    unroller = AutoregressiveStepper(
      stepper=stepper,
      dt=(dataset.dt),
      tau_max=(tau_ratio_max * dataset.dt),
    )
    apply_jit = jax.jit(unroller.unroll, static_argnums=(2,))
  else:
    apply_jit = jax.jit(stepper.apply)

  # Autoregressive rollout
  u_prd = []
  for i in range(repeats):
    t0 = time()
    subkey, key = jax.random.split(key)
    if dataset.time_dependent:
      u_prd.append(
        _get_estimations_in_batches(
          apply_fn=apply_jit,
          transform=(lambda b: _change_resolution(b, train_flags['space_downsample_factor'])),
          key=subkey,
        )[:, [0, IDX_FN, -1]]
      )
    else:
      u_prd.append(
        _get_estimations_in_batches(
          apply_fn=apply_jit,
          transform=(lambda b: _change_resolution(b, train_flags['space_downsample_factor'])),
          key=subkey,
        )
      )
    _print_between_dashes(f'ensemble_repeat={i} \t TIME={time()-t0 : .4f}s')
  u_prd = np.stack(u_prd)

  return u_prd

def main(argv):
  # Check the number of arguments
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Check the available devices
  # NOTE: We only support single-host training.
  with disable_logging():
    process_index = jax.process_index()
    process_count = jax.process_count()
    local_devices = jax.local_devices()
  logging.info('JAX host: %d / %d', process_index, process_count)
  logging.info('JAX local devices: %r', local_devices)
  assert process_count == 1

  # Read the arguments and check
  assert FLAGS.batch_size % NUM_DEVICES == 0
  datapath = '/'.join(FLAGS.exp.split('/')[1:-1])
  DIR = DIR_EXPERIMENTS / FLAGS.exp

  # Set the dataset
  dataset = Dataset(
    datadir=FLAGS.datadir,
    datapath=datapath,
    include_passive_variables=False,
    concatenate_coeffs=False,
    time_downsample_factor=1,
    space_downsample_factor=1,
    n_train=0,
    n_valid=0,
    n_test=FLAGS.n_test,
    preload=True,
  )
  dataset_small = Dataset(
    datadir=FLAGS.datadir,
    datapath=datapath,
    include_passive_variables=False,
    concatenate_coeffs=False,
    time_downsample_factor=1,
    space_downsample_factor=1,
    n_train=0,
    n_valid=0,
    n_test=min(4, FLAGS.n_test),
    preload=True,
  )

  # Read the stats
  with open(DIR / 'stats.pkl', 'rb') as f:
    stats = pickle.load(f)
  stats = {
      key: {
        k: jnp.array(v) if (v is not None) else None
        for k, v in val.items()
      }
      for key, val in stats.items()
    }
  # Read the configs
  with open(DIR / 'configs.json', 'rb') as f:
    configs = json.load(f)
  time_downsample_factor = configs['flags']['time_downsample_factor']
  tau_max_train = configs['flags']['tau_max']
  model_configs = configs['model_configs']
  # Read the state
  orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  mngr = orbax.checkpoint.CheckpointManager(DIR / 'checkpoints')
  best_checkpointed_step = mngr.best_step()
  ckpt = orbax_checkpointer.restore(directory=(DIR / 'checkpoints' / str(best_checkpointed_step) / 'default'))
  state = jax.tree_util.tree_map(jnp.array, ckpt['state'])

  # Set the stepper type
  if configs['flags']['stepper'] == 'out':
    stepping = OutputStepper
  elif configs['flags']['stepper'] == 'res':
    stepping = ResidualStepper
  elif configs['flags']['stepper'] == 'der':
    stepping = TimeDerivativeStepper
  else:
    raise ValueError

  # Set the model
  model = RIGNO(**model_configs)

  # Define the graph builder
  graph_builder = RegionInteractionGraphBuilder(
    periodic=dataset.metadata.periodic,
    rmesh_levels=configs['flags']['rmesh_levels'],
    subsample_factor=configs['flags']['mesh_subsample_factor'],
    overlap_factor_p2r=configs['flags']['overlap_factor_p2r'],
    overlap_factor_r2p=configs['flags']['overlap_factor_r2p'],
    node_coordinate_freqs=configs['flags']['node_coordinate_freqs'],
  )
  correction_sdsf = configs['flags']['space_downsample_factor'] / 1
  dataset.build_graphs(graph_builder, rmesh_correction_dsf=correction_sdsf)
  dataset_small.build_graphs(graph_builder, rmesh_correction_dsf=correction_sdsf)

  # Profile inference time
  # NOTE: One compilation per each profiling
  profile_inference(
    dataset=dataset,
    graph_builder=graph_builder,
    model=model,
    stepping=stepping,
    state=state,
    stats=stats,
    p_edge_masking=0,
  )
  profile_inference(
    dataset=dataset,
    graph_builder=graph_builder,
    model=model,
    stepping=stepping,
    state=state,
    stats=stats,
    p_edge_masking=.5,
  )
  profile_inference(
    dataset=dataset,
    graph_builder=graph_builder,
    model=model,
    stepping=stepping,
    state=state,
    stats=stats,
    p_edge_masking=.8,
  )
  if FLAGS.only_profile:
    return

  # Create a clean directory for tests
  DIR_TESTS = DIR / 'tests'
  DIR_FIGS = DIR_TESTS / 'figures'
  if DIR_TESTS.exists():
    shutil.rmtree(DIR_TESTS)
  DIR_TESTS.mkdir()
  DIR_FIGS.mkdir()

  # Set evaluation settings
  tau_min = 1
  tau_max = (tau_max_train + (tau_max_train > 1)) * time_downsample_factor
  taus_direct = [.5] + list(range(tau_min, tau_max + 1))
  # NOTE: One compilation per tau_rollout
  taus_rollout = [.5, 1] + [time_downsample_factor * d for d in range(1, tau_max_train+1)]
  # NOTE: Two compilations per discretization
  space_dsfs = [3, 2.5, 2, 1.7, 1.4, 1] if FLAGS.resolution else []
  noise_levels = [0, .005, .01, .02] if FLAGS.noise else []

  # Set the ground-truth trajectories
  batch_tst = next(dataset.batches(mode='test', batch_size=dataset.nums['test'], get_graphs=False))
  batch_tst_small = next(dataset_small.batches(mode='test', batch_size=dataset_small.nums['test'], get_graphs=False))

  # Get model estimations with all settings
  errors, u_prd = get_all_estimations(
    dataset=dataset,
    model=model,
    graph_builder=graph_builder,
    stepping=stepping,
    state=state,
    stats=stats,
    train_flags=configs['flags'],
    taus_direct=(taus_direct if dataset.time_dependent else []),
    taus_rollout=(taus_rollout if dataset.time_dependent else []),
    space_dsfs=space_dsfs,
    noise_levels=noise_levels,
    p_edge_masking=0,
  )

  # Plot estimation visualizations
  (DIR_FIGS / 'samples').mkdir()
  for s in range(min(4, FLAGS.n_test)):
    _batch_tst = _change_resolution(batch_tst, configs['flags']['space_downsample_factor'])
    if dataset.time_dependent:
      fig = plot_estimates(
        u_inp=_batch_tst.u[s, 0],
        u_gtr=_batch_tst.u[s, IDX_FN],
        u_prd=u_prd[s, 1],
        x_inp=_batch_tst.x[s, 0],
        x_out=_batch_tst.x[s, 0],
        symmetric=dataset.metadata.signed['u'],
        names=dataset.metadata.names['u'],
        domain=dataset.metadata.domain_x,
      )
      fig.savefig(DIR_FIGS / 'samples' / f'rollout-fn-s{s:02d}.png', dpi=300, bbox_inches='tight')
      plt.close(fig)
      fig = plot_estimates(
        u_inp=_batch_tst.u[s, 0],
        u_gtr=_batch_tst.u[s, -1],
        u_prd=u_prd[s, -1],
        x_inp=_batch_tst.x[s, 0],
        x_out=_batch_tst.x[s, 0],
        symmetric=dataset.metadata.signed['u'],
        names=dataset.metadata.names['u'],
        domain=dataset.metadata.domain_x,
      )
      fig.savefig(DIR_FIGS / 'samples' / f'rollout-ex-s{s:02d}.png', dpi=300, bbox_inches='tight')
      plt.close(fig)
    else:
      fig = plot_estimates(
        u_inp=_batch_tst.c[s, 0],
        u_gtr=_batch_tst.u[s, 0],
        u_prd=u_prd[s, 0],
        x_inp=_batch_tst.x[s, 0],
        x_out=_batch_tst.x[s, 0],
        symmetric=dataset.metadata.signed['u'],
        names=dataset.metadata.names['u'],
        domain=dataset.metadata.domain_x,
      )
      fig.savefig(DIR_FIGS / 'samples' / f's{s:02d}.png', dpi=300, bbox_inches='tight')
      plt.close(fig)

  # Store the errors
  with open(DIR_TESTS / 'errors.json', 'w') as f:
    json.dump(obj=errors, fp=f)

  # Print minimum errors
  if dataset.time_dependent:
    l1_final = min([errors['_l1']['tau']['rollout'][tau][IDX_FN] for tau in taus_rollout])
    l2_final = min([errors['_l2']['tau']['rollout'][tau][IDX_FN] for tau in taus_rollout])
    l1_extra = min([errors['_l2']['tau']['rollout'][tau][-1] for tau in taus_rollout])
    l2_extra = min([errors['_l2']['tau']['rollout'][tau][-1] for tau in taus_rollout])
    _print_between_dashes(f'ERROR AT t={IDX_FN} \t _l1: {l1_final : .2f}% \t _l2: {l2_final : .2f}%')
    _print_between_dashes(f'ERROR AT t={dataset.shape[1]-1} \t _l1: {l1_extra : .2f}% \t _l2: {l2_extra : .2f}%')
  else:
    l1 = errors['_l1']['disc'][0][0]
    l2 = errors['_l2']['disc'][0][0]
    _print_between_dashes(f'ERROR \t _l1: {l1 : .2f}% \t _l2: {l2 : .2f}%')


  # Plot the errors and store the plots
  if dataset.time_dependent:
    (DIR_FIGS / 'errors').mkdir()
    def errors_to_df(_errors):
      df = pd.DataFrame(_errors)
      df['t'] = df.index
      df = df.melt(id_vars=['t'], value_name='error')
      return df
    # Set which errors to plot
    errors_plot = errors['_l1']
    # Temporal continuity
    g = plot_error_vs_time(
      df=errors_to_df(errors_plot['tau']['direct']),
      idx_fn=IDX_FN,
      variable_title='$\\dfrac{\\tau}{\Delta t}$',
    )
    g.figure.savefig(DIR_FIGS / 'errors' / 'tau-direct.png', dpi=300, bbox_inches='tight')
    plt.close(g.figure)
    g = plot_error_vs_time(
      df=errors_to_df(errors_plot['tau']['rollout']),
      idx_fn=IDX_FN,
      variable_title='$\\dfrac{\\tau_{max}}{\Delta t}$',
    )
    g.figure.savefig(DIR_FIGS / 'errors' / 'tau-rollout.png', dpi=300, bbox_inches='tight')
    plt.close(g.figure)
    # Noise control
    g = plot_error_vs_time(
      df=errors_to_df(errors_plot['noise']['direct']),
      idx_fn=IDX_FN,
      variable_title='$\\dfrac{\\sigma_{noise}}{\\sigma_{data}}$',
    )
    g.figure.savefig(DIR_FIGS / 'errors' / 'noise-direct.png', dpi=300, bbox_inches='tight')
    plt.close(g.figure)
    g = plot_error_vs_time(
      df=errors_to_df(errors_plot['noise']['rollout']),
      idx_fn=IDX_FN,
      variable_title='$\\dfrac{\\sigma_{noise}}{\\sigma_{data}}$',
    )
    g.figure.savefig(DIR_FIGS / 'errors' / 'noise-rollout.png', dpi=300, bbox_inches='tight')
    plt.close(g.figure)
    # Discretization invariance
    g = plot_error_vs_time(
      df=errors_to_df(errors_plot['disc']['direct']),
      idx_fn=IDX_FN,
      variable_title='Discretization',
    )
    g.figure.savefig(DIR_FIGS / 'errors' / 'discretization-direct.png', dpi=300, bbox_inches='tight')
    plt.close(g.figure)
    g = plot_error_vs_time(
      df=errors_to_df(errors_plot['disc']['rollout']),
      idx_fn=IDX_FN,
      variable_title='Discretization',
    )
    g.figure.savefig(DIR_FIGS / 'errors' / 'discretization-rollout.png', dpi=300, bbox_inches='tight')
    plt.close(g.figure)
    # Resolution invariance
    g = plot_error_vs_time(
      df=errors_to_df(errors_plot['dsf']['direct']),
      idx_fn=IDX_FN,
      variable_title='DSF',
    )
    g.figure.savefig(DIR_FIGS / 'errors' / 'resolution-direct.png', dpi=300, bbox_inches='tight')
    plt.close(g.figure)
    g = plot_error_vs_time(
      df=errors_to_df(errors_plot['dsf']['rollout']),
      idx_fn=IDX_FN,
      variable_title='DSF',
    )
    g.figure.savefig(DIR_FIGS / 'errors' / 'resolution-rollout.png', dpi=300, bbox_inches='tight')
    plt.close(g.figure)

  # Get ensemble estimations with the default settings
  # NOTE: One compilation
  if FLAGS.ensemble:
    key = jax.random.PRNGKey(45)
    subkey, key = jax.random.split(key)
    u_prd_ensemble = get_ensemble_estimations(
      repeats=20,
      dataset=dataset_small,
      model=model,
      graph_builder=graph_builder,
      stepping=stepping,
      state=state,
      stats=stats,
      train_flags=configs['flags'],
      tau_ratio_max=time_downsample_factor,
      p_edge_masking=0.5,
      key=subkey,
    )

    # Plot ensemble statistics
    (DIR_FIGS / 'ensemble').mkdir()
    _batch_tst_small = _change_resolution(batch_tst_small, configs['flags']['space_downsample_factor'])
    for s in range(min(4, FLAGS.n_test)):
      if dataset.time_dependent:
        fig = plot_ensemble(
          u_gtr=_batch_tst_small.u[:, [0, IDX_FN, -1]],
          u_ens=u_prd_ensemble,
          x=_batch_tst_small.x[s, 0],
          idx_out=1,
          idx_s=s,
          symmetric=dataset_small.metadata.signed['u'],
          names=dataset_small.metadata.names['u'],
          domain=dataset.metadata.domain_x,
        )
        fig.savefig(DIR_FIGS / 'ensemble' / f'rollout-fn-s{s:02d}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        fig = plot_ensemble(
          u_gtr=_batch_tst_small.u[:, [0, IDX_FN, -1]],
          u_ens=u_prd_ensemble,
          x=_batch_tst_small.x[s, 0],
          idx_out=-1,
          idx_s=s,
          symmetric=dataset_small.metadata.signed['u'],
          names=dataset_small.metadata.names['u'],
          domain=dataset.metadata.domain_x,
        )
        fig.savefig(DIR_FIGS / 'ensemble' / f'rollout-ex-s{s:02d}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
      else:
        fig = plot_ensemble(
          u_gtr=_batch_tst_small.u,
          u_ens=u_prd_ensemble,
          x=_batch_tst_small.x[s, 0],
          idx_out=0,
          idx_s=s,
          symmetric=dataset_small.metadata.signed['u'],
          names=dataset_small.metadata.names['u'],
          domain=dataset.metadata.domain_x,
        )
        fig.savefig(DIR_FIGS / 'ensemble' / f's{s:02d}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)


  _print_between_dashes('DONE')

if __name__ == '__main__':
  logging.set_verbosity('info')
  define_flags()
  app.run(main)
