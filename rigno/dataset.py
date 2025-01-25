"""Classes and metadata for reading the datasets."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Sequence, NamedTuple, Literal

from flax.typing import PRNGKey
import h5py
import jax
import jax.lax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np

from rigno.models.rigno import (
  RegionInteractionGraphMetadata,
  RegionInteractionGraphSet,
  RegionInteractionGraphBuilder)
from rigno.utils import Array


@dataclass
class Metadata:
  """Holder of metadata for a dataset."""

  periodic: bool
  group_u: str
  group_c: str
  group_x: str
  type: Literal['poseidon', 'rigno']
  fix_x: bool
  domain_x: tuple[Sequence[int], Sequence[int]]
  domain_t: tuple[int, int]
  active_variables: Sequence[int]  # Index of variables in input/output
  chunked_variables: Sequence[int]  # Index of variable groups
  num_variable_chunks: int  # Number of variable chunks
  signed: dict[str, Union[bool, Sequence[bool]]]
  names: dict[str, Sequence[str]]
  global_mean: Sequence[float]
  global_std: Sequence[float]

ACTIVE_VARS_NS = [0, 1]
ACTIVE_VARS_CE = [0, 1, 2, 3]
ACTIVE_VARS_GCE = [0, 1, 2, 3, 5]
ACTIVE_VARS_RD = [0]
ACTIVE_VARS_WE = [0]
ACTIVE_VARS_PE = [0]

CHUNKED_VARS_NS = [0, 0]
CHUNKED_VARS_CE = [0, 1, 1, 2, 3]
CHUNKED_VARS_GCE = [0, 1, 1, 2, 3, 4]
CHUNKED_VARS_RD = [0]
CHUNKED_VARS_WE = [0]
CHUNKED_VARS_PE = [0]

SIGNED_NS = {'u': [True, True], 'c': None}
SIGNED_CE = {'u': [False, True, True, False, False], 'c': None}
SIGNED_GCE = {'u': [False, True, True, False, False, False], 'c': None}
SIGNED_RD = {'u': [True], 'c': None}
SIGNED_WE = {'u': [True], 'c': [False]}
SIGNED_PE = {'u': [True], 'c': [True]}

NAMES_NS = {'u': ['$v_x$', '$v_y$'], 'c': None}
NAMES_CE = {'u': ['$\\rho$', '$v_x$', '$v_y$', '$p$'], 'c': None}
NAMES_GCE = {'u': ['$\\rho$', '$v_x$', '$v_y$', '$p$', 'E', '$\\phi$'], 'c': None}
NAMES_RD = {'u': ['$u$'], 'c': None}
NAMES_WE = {'u': ['$u$'], 'c': ['$c$']}
NAMES_PE = {'u': ['$u$'], 'c': ['$f$']}

DATASET_METADATA = {
  # incompressible_fluids: [velocity, velocity]
  'incompressible_fluids/brownian_bridge': Metadata(
    periodic=True,
    group_u='velocity',
    group_c=None,
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
  'incompressible_fluids/gaussians': Metadata(
    periodic=True,
    group_u='velocity',
    group_c=None,
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
  'incompressible_fluids/pwc': Metadata(
    periodic=True,
    group_u='velocity',
    group_c=None,
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
  'incompressible_fluids/shear_layer': Metadata(
    periodic=True,
    group_u='velocity',
    group_c=None,
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
  'incompressible_fluids/sines': Metadata(
    periodic=True,
    group_u='velocity',
    group_c=None,
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
  'incompressible_fluids/vortex_sheet': Metadata(
    periodic=True,
    group_u='velocity',
    group_c=None,
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
  # compressible_flow: [density, velocity, velocity, pressure, energy]
  'compressible_flow/gauss': Metadata(
    periodic=True,
    group_u='data',
    group_c=None,
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_CE,
    chunked_variables=CHUNKED_VARS_CE,
    num_variable_chunks=len(set(CHUNKED_VARS_CE)),
    signed=SIGNED_CE,
    names=NAMES_CE,
    global_mean=[0.80, 0., 0., 2.513],
    global_std=[0.31, 0.391, 0.356, 0.185],
  ),
  'compressible_flow/kh': Metadata(
    periodic=True,
    group_u='data',
    group_c=None,
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_CE,
    chunked_variables=CHUNKED_VARS_CE,
    num_variable_chunks=len(set(CHUNKED_VARS_CE)),
    signed=SIGNED_CE,
    names=NAMES_CE,
    global_mean=[0.80, 0., 0., 1.0],
    global_std=[0.31, 0.391, 0.356, 0.185],
  ),
  'compressible_flow/richtmyer_meshkov': Metadata(
    periodic=True,
    group_u='solution',
    group_c=None,
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 2),
    fix_x=True,
    active_variables=ACTIVE_VARS_CE,
    chunked_variables=CHUNKED_VARS_CE,
    num_variable_chunks=len(set(CHUNKED_VARS_CE)),
    signed=SIGNED_CE,
    names=NAMES_CE,
    global_mean=[1.1964245, -7.164812e-06, 2.8968952e-06, 1.5648036],
    global_std=[0.5543239, 0.24304213, 0.2430597, 0.89639103],
  ),
  'compressible_flow/riemann': Metadata(
    periodic=True,
    group_u='data',
    group_c=None,
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_CE,
    chunked_variables=CHUNKED_VARS_CE,
    num_variable_chunks=len(set(CHUNKED_VARS_CE)),
    signed=SIGNED_CE,
    names=NAMES_CE,
    global_mean=[0.80, 0., 0., 0.215],
    global_std=[0.31, 0.391, 0.356, 0.185],
  ),
  'compressible_flow/riemann_curved': Metadata(
    periodic=True,
    group_u='data',
    group_c=None,
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_CE,
    chunked_variables=CHUNKED_VARS_CE,
    num_variable_chunks=len(set(CHUNKED_VARS_CE)),
    signed=SIGNED_CE,
    names=NAMES_CE,
    global_mean=[0.80, 0., 0., 0.553],
    global_std=[0.31, 0.391, 0.356, 0.185],
  ),
  'compressible_flow/riemann_kh': Metadata(
    periodic=True,
    group_u='data',
    group_c=None,
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_CE,
    chunked_variables=CHUNKED_VARS_CE,
    num_variable_chunks=len(set(CHUNKED_VARS_CE)),
    signed=SIGNED_CE,
    names=NAMES_CE,
    global_mean=[0.80, 0., 0., 1.33],
    global_std=[0.31, 0.391, 0.356, 0.185],
  ),
  'compressible_flow/gravity/rayleigh_taylor': Metadata(
    periodic=True,
    group_u='solution',
    group_c=None,
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 5),
    fix_x=True,
    active_variables=ACTIVE_VARS_GCE,
    chunked_variables=CHUNKED_VARS_GCE,
    num_variable_chunks=len(set(CHUNKED_VARS_GCE)),
    signed=SIGNED_GCE,
    names=NAMES_GCE,
    global_mean=[0.8970493, 4.0316996e-13, -1.3858967e-13, 0.7133829, -1.7055787],
    global_std=[0.12857835, 0.014896976, 0.014896975, 0.21293919, 0.40131348],
  ),
  # reaction_diffusion
  'reaction_diffusion/allen_cahn': Metadata(
    periodic=False,
    group_u='solution',
    group_c=None,
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 0.0002),
    fix_x=True,
    active_variables=ACTIVE_VARS_RD,
    chunked_variables=CHUNKED_VARS_RD,
    num_variable_chunks=len(set(CHUNKED_VARS_RD)),
    signed=SIGNED_RD,
    names=NAMES_RD,
    global_mean=[0.002484262],
    global_std=[0.65351176],
  ),
  # wave_equation
  'wave_equation/seismic_20step': Metadata(
    periodic=False,
    group_u='solution',
    group_c='c',
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_WE,
    chunked_variables=CHUNKED_VARS_WE,
    num_variable_chunks=len(set(CHUNKED_VARS_WE)),
    signed=SIGNED_WE,
    names=NAMES_WE,
    global_mean=[0.03467443221585092],
    global_std=[0.10442421752963911],
  ),
  'wave_equation/gaussians_15step': Metadata(
    periodic=False,
    group_u='solution',
    group_c='c',
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_WE,
    chunked_variables=CHUNKED_VARS_WE,
    num_variable_chunks=len(set(CHUNKED_VARS_WE)),
    signed=SIGNED_WE,
    names=NAMES_WE,
    global_mean=[0.0334376316],
    global_std=[0.1171879068],
  ),
  # poisson_equation
  'poisson_equation/sines': Metadata(
    periodic=False,
    group_u='solution',
    group_c='source',
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=None,
    fix_x=True,
    active_variables=ACTIVE_VARS_PE,
    chunked_variables=CHUNKED_VARS_PE,
    num_variable_chunks=len(set(CHUNKED_VARS_PE)),
    signed=SIGNED_PE,
    names=NAMES_PE,
    global_mean=[0.0005603458434937093],
    global_std=[0.02401226126952699],
  ),
  'poisson_equation/chebyshev': Metadata(
    periodic=False,
    group_u='solution',
    group_c='source',
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=None,
    fix_x=True,
    active_variables=ACTIVE_VARS_PE,
    chunked_variables=CHUNKED_VARS_PE,
    num_variable_chunks=len(set(CHUNKED_VARS_PE)),
    signed=SIGNED_PE,
    names=NAMES_PE,
    global_mean=[0.0005603458434937093],
    global_std=[0.02401226126952699],
  ),
  'poisson_equation/pwc': Metadata(
    periodic=False,
    group_u='solution',
    group_c='source',
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=None,
    fix_x=True,
    active_variables=ACTIVE_VARS_PE,
    chunked_variables=CHUNKED_VARS_PE,
    num_variable_chunks=len(set(CHUNKED_VARS_PE)),
    signed=SIGNED_PE,
    names=NAMES_PE,
    global_mean=[0.0005603458434937093],
    global_std=[0.02401226126952699],
  ),
  'poisson_equation/gaussians': Metadata(
    periodic=False,
    group_u='solution',
    group_c='source',
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=None,
    fix_x=True,
    active_variables=ACTIVE_VARS_PE,
    chunked_variables=CHUNKED_VARS_PE,
    num_variable_chunks=len(set(CHUNKED_VARS_PE)),
    signed=SIGNED_PE,
    names=NAMES_PE,
    global_mean=[0.0005603458434937093],
    global_std=[0.02401226126952699],
  ),
  # steady Euler
  'rigno-unstructured/airfoil_grid': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x=None,
    type='poseidon',
    domain_x=([-.75, -.75], [1.75, 1.75]),
    domain_t=None,
    fix_x=True,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [False], 'c': [False]},
    names={'u': ['$\\rho$'], 'c': ['$d$']},
    global_mean=[0.92984116],
    global_std=[0.10864315],
  ),
  # rigno-unstructured
  'rigno-unstructured/airfoil_li': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='rigno',
    domain_x=([-1, -1], [2, 1]),
    domain_t=None,
    fix_x=False,
    active_variables=[0],  # Only the density
    chunked_variables=[0, 1, 1, 2, 3],
    num_variable_chunks=4,
    signed={'u': [False, True, True, False, False], 'c': [False]},
    names={'u': ['$\\rho$', '$v_x$', '$v_y$', '$p$', '$Ma$'], 'c': ['$d$']},
    global_mean=[0.9637927979586245],
    global_std=[0.11830822800242624],
  ),
  'rigno-unstructured/airfoil_li_large': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='rigno',
    domain_x=([-3, -3], [+5, +3]),
    domain_t=None,
    fix_x=False,
    active_variables=[0],  # Only the density
    chunked_variables=[0, 1, 1, 2, 3],
    num_variable_chunks=4,
    signed={'u': [False, True, True, False, False], 'c': [False]},
    names={'u': ['$\\rho$', '$v_x$', '$v_y$', '$p$', '$Ma$'], 'c': ['$d$']},
    global_mean=[0.9637927979586245],
    global_std=[0.11830822800242624],
  ),
  'rigno-unstructured/elasticity': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=None,
    fix_x=False,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [False], 'c': [False]},
    names={'u': ['$\\sigma$'], 'c': ['$d$']},
    global_mean=[187.477],
    global_std=[127.046],
  ),
  'rigno-unstructured/poisson_c_sines': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='rigno',
    domain_x=([-.5, -.5], [1.5, 1.5]),
    domain_t=None,
    fix_x=True,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [True], 'c': [True]},
    names={'u': ['$u$'], 'c': ['$f$']},
    global_mean=[0.],
    global_std=[0.00064911455],
  ),
  'rigno-unstructured/wave_c_sines': Metadata(
    periodic=False,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([-.5, -.5], [1.5, 1.5]),
    domain_t=(0, 0.1),
    fix_x=True,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [True], 'c': None},
    names={'u': ['$u$'], 'c': None},
    global_mean=[0.],
    global_std=[0.011314605],
  ),
  'rigno-unstructured/wave_c_sines_uv': Metadata(
    periodic=False,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([-.5, -.5], [1.5, 1.5]),
    domain_t=(0, 0.1),
    fix_x=True,
    active_variables=[0, 1],
    chunked_variables=[0, 1],
    num_variable_chunks=2,
    signed={'u': [True, True], 'c': None},
    names={'u': ['$u$', '$\\partial_t u$'], 'c': None},
    global_mean=[0., 0.],
    global_std=[0.004625, 0.3249],
  ),
  'rigno-unstructured/heat_l_sines': Metadata(
    periodic=False,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0., 0.], [1., 1.]),
    domain_t=(0, 0.002),
    fix_x=True,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [True], 'c': None},
    names={'u': ['$u$'], 'c': None},
    global_mean=[-0.009399102],
    global_std=[0.020079814],
  ),
  'rigno-unstructured/NS-Gauss': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
  'rigno-unstructured/NS-PwC': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
  'rigno-unstructured/NS-SL': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
  'rigno-unstructured/NS-SVS': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
  'rigno-unstructured/CE-Gauss': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_CE,
    chunked_variables=CHUNKED_VARS_CE,
    num_variable_chunks=len(set(CHUNKED_VARS_CE)),
    signed=SIGNED_CE,
    names=NAMES_CE,
    global_mean=[0.80, 0., 0., 2.513],
    global_std=[0.31, 0.391, 0.356, 0.185],
  ),
  'rigno-unstructured/CE-RP': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_CE,
    chunked_variables=CHUNKED_VARS_CE,
    num_variable_chunks=len(set(CHUNKED_VARS_CE)),
    signed=SIGNED_CE,
    names=NAMES_CE,
    global_mean=[0.80, 0., 0., 0.215],
    global_std=[0.31, 0.391, 0.356, 0.185],
  ),
  'rigno-unstructured/ACE': Metadata(
    periodic=False,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 0.0002),
    fix_x=True,
    active_variables=ACTIVE_VARS_RD,
    chunked_variables=CHUNKED_VARS_RD,
    num_variable_chunks=len(set(CHUNKED_VARS_RD)),
    signed=SIGNED_RD,
    names=NAMES_RD,
    global_mean=[0.002484262],
    global_std=[0.65351176],
  ),
  'rigno-unstructured/Wave-Layer': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_WE,
    chunked_variables=CHUNKED_VARS_WE,
    num_variable_chunks=len(set(CHUNKED_VARS_WE)),
    signed=SIGNED_WE,
    names=NAMES_WE,
    global_mean=[0.03467443221585092],
    global_std=[0.10442421752963911],
  ),
  'rigno-unstructured/Poisson-Gauss': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=None,
    fix_x=True,
    active_variables=ACTIVE_VARS_PE,
    chunked_variables=CHUNKED_VARS_PE,
    num_variable_chunks=len(set(CHUNKED_VARS_PE)),
    signed=SIGNED_PE,
    names=NAMES_PE,
    global_mean=[0.0005603458434937093],
    global_std=[0.02401226126952699],
  ),
}

class Batch(NamedTuple):
  """A batch of data samples."""

  u: Array
  c: Union[None, Array]
  x: Array
  t: Union[None, Array]
  g: Union[None, Sequence[RegionInteractionGraphSet]]

  @property
  def shape(self) -> tuple:
    return self.u.shape

  def unravel(self) -> tuple:
    return (self.u, self.c, self.x, self.t)

  def __len__(self) -> int:
    return self.shape[0]

class Dataset:
  """Class for reading a dataset."""

  def __init__(self,
    datadir: str,
    datapath: str,
    include_passive_variables: bool = False,
    concatenate_coeffs: bool = False,
    time_cutoff_idx: int = None,
    time_downsample_factor: int = 1,
    space_downsample_factor: float = 1.,
    n_train: int = 0,
    n_valid: int = 0,
    n_test: int = 0,
    preload: bool = False,
    key: PRNGKey = None,
  ):
    """
    Args:
        datadir: Directory of the datasets.
        datapath: Relative path of the dataset in the data directory.
        include_passive_variables: If True, passive variables of the dataset are returned. Defaults to False.
        concatenate_coeffs: If True, the coefficients are concatenated to the main function. Defaults to False.
        time_cutoff_idx: Time index where the trajectories are cut off. Defaults to None.
        time_downsample_factor: A factor for downsampling in the time axis. Defaults to 1.
        space_downsample_factor: A factor for downsampling in the space axis (in each direction). Defaults to 1.
        n_train: Number of training samples. Defaults to 0.
        n_valid: Number of validation samples. Defaults to 0.
        n_test: Number of test samples. Defaults to 0.
        preload: If True, the whole dataset is loaded in CPU memory. Defaults to False.
        key: Random number generator key for random processes such as downsampling and shuffles. Defaults to None.
    """

    # Set attributes
    self.key = key if (key is not None) else jax.random.PRNGKey(0)
    self.metadata = deepcopy(DATASET_METADATA[datapath])
    self.preload = preload
    self.concatenate_coeffs = concatenate_coeffs
    self.time_cutoff_idx = time_cutoff_idx
    self.time_downsample_factor = time_downsample_factor
    self.space_downsample_factor = space_downsample_factor

    # Modify metadata
    if not include_passive_variables:
      self.metadata.names['u'] = [self.metadata.names['u'][v] for v in self.metadata.active_variables]
      self.metadata.signed['u'] = [self.metadata.signed['u'][v] for v in self.metadata.active_variables]
      self.metadata.chunked_variables = [self.metadata.chunked_variables[v] for v in self.metadata.active_variables]
      self.metadata.chunked_variables = [v - min(self.metadata.chunked_variables) for v in self.metadata.chunked_variables]
      self.metadata.num_variable_chunks = len(set(self.metadata.chunked_variables))
    if self.concatenate_coeffs and self.metadata.group_c:
      self.metadata.names['u'] += self.metadata.names['c']
      self.metadata.signed['u'] += self.metadata.signed['c']

    # Set data attributes
    self.u, self.c, self.x, self.t = None, None, None, None
    self.rigs: RegionInteractionGraphMetadata = None
    self.data_group = self.metadata.group_u
    self.coeff_group = self.metadata.group_c
    self.coords_group = self.metadata.group_x
    self.reader = h5py.File(Path(datadir) / f'{datapath}.nc', 'r')
    self.idx_vars = (None if include_passive_variables
      else self.metadata.active_variables)
    self.length = ((n_train + n_valid + n_test) if self.preload
      else self.reader[self.data_group].shape[0])
    self.sample = self._fetch(0)
    self.shape = self.sample.shape
    if self.time_dependent:
      self.dt = (self.sample.t[0, 1] - self.sample.t[0, 0]).item() # NOTE: Assuming fix dt

    # Check sample dimensions
    for arr in self.sample.unravel():
      if arr is not None:
        assert arr.ndim == 4

    # Split the dataset
    assert (n_train + n_valid + n_test) <= self.length
    self.nums = {'train': n_train, 'valid': n_valid, 'test': n_test}
    self.idx_modes = {
      # First n_train samples
      'train': jax.random.permutation(self.key, n_train),
      # First n_valid samples after the training samples
      'valid': np.arange(n_train, (n_train + n_valid)),
      # Last n_test samples
      'test': np.arange((self.length - n_test), self.length),
    }

    # Instantiate the dataset stats
    self.stats = {
      'u': {'mean': None, 'std': None},
      'c': {'mean': None, 'std': None},
      'x': {
        'min': np.array(self.metadata.domain_x[0]).reshape(1, 1, 1, -1),
        'max': np.array(self.metadata.domain_x[1]).reshape(1, 1, 1, -1),
      },
      't': {
        'min': np.array(self.metadata.domain_t[0]).reshape(1, 1, 1, 1)
          if self.time_dependent else None,
        'max': np.array(self.metadata.domain_t[1]).reshape(1, 1, 1, 1)
          if self.time_dependent else None,
      },
      'res': {'mean': None, 'std': None},
      'der': {'mean': None, 'std': None},
    }

    # Load the data
    if self.preload:
      _len_dataset = self.reader[self.data_group].shape[0]
      u_trn = self.reader[self.data_group][np.arange(n_train)]
      u_val = self.reader[self.data_group][np.arange(n_train, (n_train + n_valid))]
      u_tst = self.reader[self.data_group][np.arange((_len_dataset - n_test), (_len_dataset))]
      self.u = np.concatenate([u_trn, u_val, u_tst], axis=0)
      if self.coeff_group is not None:
        c_trn = self.reader[self.coeff_group][np.arange(n_train)]
        c_val = self.reader[self.coeff_group][np.arange(n_train, (n_train + n_valid))]
        c_tst = self.reader[self.coeff_group][np.arange((_len_dataset - n_test), (_len_dataset))]
        self.c = np.concatenate([c_trn, c_val, c_tst], axis=0)
      if self.coords_group is not None:
        if self.metadata.fix_x:
          self.x = self.reader[self.coords_group]
        else:
          x_trn = self.reader[self.coords_group][np.arange(n_train)]
          x_val = self.reader[self.coords_group][np.arange(n_train, (n_train + n_valid))]
          x_tst = self.reader[self.coords_group][np.arange((_len_dataset - n_test), (_len_dataset))]
          self.x = np.concatenate([x_trn, x_val, x_tst], axis=0)

  @property
  def time_dependent(self) -> bool:
    """Wether the dataset is time-dependent."""

    return self.metadata.domain_t is not None

  def compute_stats(self, residual_steps: int = 0) -> None:
    """Computes the statistics of the dataset based on the training samples."""

    # Check inputs
    assert residual_steps >= 0
    assert residual_steps < self.shape[1]

    # Get all trajectories
    batch = self.train(np.arange(self.nums['train']))
    u, c, _, t = batch.unravel()

    # Compute statistics of solutions and coefficients
    self.stats['u']['mean'] = np.mean(u, axis=(0, 1, 2), keepdims=True)
    self.stats['u']['std'] = np.std(u, axis=(0, 1, 2), keepdims=True)
    if c is not None:
      self.stats['c']['mean'] = np.mean(c, axis=(0, 1, 2), keepdims=True)
      self.stats['c']['std'] = np.std(c, axis=(0, 1, 2), keepdims=True)

    # Compute statistics of the residuals and time derivatives
    if self.time_dependent:
      _get_res = lambda s, trj: (trj[:, (s):] - trj[:, :-(s)])
      residuals = []
      derivatives = []
      for s in range(1, residual_steps+1):
        res = _get_res(s, u)
        tau = _get_res(s, t)
        residuals.append(res)
        derivatives.append(res / tau)
      residuals = np.concatenate(residuals, axis=1)
      derivatives = np.concatenate(derivatives, axis=1)
      self.stats['res']['mean'] = np.mean(residuals, axis=(0, 1, 2), keepdims=True)
      self.stats['res']['std'] = np.std(residuals, axis=(0, 1, 2), keepdims=True)
      self.stats['der']['mean'] = np.mean(derivatives, axis=(0, 1, 2), keepdims=True)
      self.stats['der']['std'] = np.std(derivatives, axis=(0, 1, 2), keepdims=True)

  def build_graphs(self, builder: RegionInteractionGraphBuilder, rmesh_correction_dsf: int = 1, key: PRNGKey = None) -> None:
    """Builds RIGNO graphs for all samples and stores them in the object."""
    # NOTE: Each graph takes about 3 MB and 2 seconds to build.
    # It can cause memory issues for large datasets.

    # NOTE: It is important to do the rmesh sub-sampling with a different key each time
    # Otherwise, for some datasets, the rmeshes can end up being similar
    if key is None:
      key = jax.random.PRNGKey(0)

    # Build graph metadata with potentially different number of edges
    # NOTE: Stores all graphs in memory one by one
    metadata = []
    num_p2r_edges = 0
    num_r2r_edges = 0
    num_r2p_edges = 0
    if self.rigs is not None:
      # NOTE: Use the old number of edges in order to avoid re-compilation
      num_p2r_edges = self.rigs.p2r_edge_indices.shape[1]
      num_r2r_edges = self.rigs.r2r_edge_indices.shape[1]
      if self.rigs.r2p_edge_indices is not None:
        num_r2p_edges = self.rigs.r2p_edge_indices.shape[1]
    for mode in ['train', 'valid', 'test']:
      if not self.nums[mode] > 0: continue
      batch = self._fetch_mode(idx=np.arange(self.nums[mode]), mode=mode)
      # Loop over all coordinates in the batch
      # NOTE: Assuming constant x in time
      for x in batch.x[:, 0]:
        key, subkey = jax.random.split(key)
        m = builder.build_metadata(x_inp=x, x_out=x, domain=np.array(self.metadata.domain_x), rmesh_correction_dsf=rmesh_correction_dsf, key=subkey)
        metadata.append(m)
        # Store the maximum number of edges
        if self.rigs is None:
          num_p2r_edges = max(num_p2r_edges, m.p2r_edge_indices.shape[1])
          num_r2r_edges = max(num_r2r_edges, m.r2r_edge_indices.shape[1])
          if m.r2p_edge_indices is not None:
            num_r2p_edges = max(num_r2p_edges, m.r2p_edge_indices.shape[1])
        # Break the loop if the coordinates are fixed on the batch axis
        if self.metadata.fix_x:
          break
      # Break the loop if the coordinates are fixed on the batch axis
      if self.metadata.fix_x:
        break

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
    # NOTE: This line duplicates the memory needed for storing the graphs
    # TODO: Make the concatenation memory efficient
    ## One way is to add another loop and write the graphs one by one in-place in the concatenated array
    self.rigs = tree.tree_map(lambda *v: jnp.concatenate(v), *metadata)

  def _fetch(self, idx: Union[int, Sequence], get_graphs: bool = True) -> Batch:
    """Fetches a sample from the dataset, given its global index."""

    # Check inputs
    if isinstance(idx, int):
      idx = [idx]

    # Get u
    if self.u is not None:
      u = self.u[np.sort(idx)]
    else:
      u = self.reader[self.data_group][np.sort(idx)]

    # Get c
    if self.coeff_group is not None:
      if self.c is not None:
        c = self.c[np.sort(idx)]
      else:
        c = self.reader[self.coeff_group][np.sort(idx)]
    else:
      c = None

    # Get graphs
    if (self.rigs is not None) and get_graphs:
      g = tree.tree_map(lambda v: v[np.sort(idx)], self.rigs)
    else:
      g = None

    if self.metadata.type == 'poseidon':
      # Re-arrange u
      if len(u.shape) == 5:  # NOTE: Multi-variable datasets
        u = np.moveaxis(u, source=(2, 3, 4), destination=(4, 2, 3))
      elif len(u.shape) == 4:  # NOTE: Single-variable datasets
        u = np.expand_dims(u, axis=-1)
      elif len(u.shape) == 3:  # NOTE: Single-variable time-independent datasets
        u = np.expand_dims(u, axis=(1, -1))
      # Re-arrange c
      if c is not None:
        c = np.expand_dims(c, axis=(1, 4))
        c = np.tile(c, reps=(1, u.shape[1], 1, 1, 1))

      # Define spatial coordinates
      assert self.coords_group is None
      _xv = np.linspace(self.metadata.domain_x[0][0], self.metadata.domain_x[1][0], u.shape[2], endpoint=(not self.metadata.periodic))
      _yv = np.linspace(self.metadata.domain_x[0][1], self.metadata.domain_x[1][1], u.shape[3], endpoint=(not self.metadata.periodic))
      _x, _y = np.meshgrid(_xv, _yv)
      # Align the dimensions
      _x = _x.reshape(1, 1, -1, 1)
      _y = _y.reshape(1, 1, -1, 1)
      # Concatenate the coordinates
      x = np.concatenate([_x, _y], axis=3)
      # Repeat along sample and time axes
      assert self.metadata.fix_x
      x = np.tile(x, reps=(u.shape[0], u.shape[1], 1, 1))

      # Flatten the trajectory
      u = u.reshape(u.shape[0], u.shape[1], (u.shape[2] * u.shape[3]), -1)
      if c is not None:
        c = c.reshape(u.shape[0], u.shape[1], (u.shape[2] * u.shape[3]), -1)

      # Define temporal coordinates
      if self.metadata.domain_t is not None:
        t = np.linspace(*self.metadata.domain_t, u.shape[1], endpoint=True)
        t = t.reshape(1, -1, 1, 1)
        # Repeat along sample trajectory
        t = np.tile(t, reps=(u.shape[0], 1, 1, 1))
      else:
        t = None

    elif self.metadata.type == 'rigno':
      # Read spatial coordinates
      assert self.coords_group is not None
      if self.x is not None:
        x = self.x if self.metadata.fix_x else self.x[np.sort(idx)]
      else:
        x = self.reader[self.coords_group] if self.metadata.fix_x else self.reader[self.coords_group][np.sort(idx)]
      # Repeat along the time axis
      # NOTE: the coordinates are assumed to be constant in time
      assert x.shape[1] == 1
      x = np.tile(x, reps=(1, u.shape[1], 1, 1))
      # Repeat along the batch axis
      if self.metadata.fix_x:
        assert x.shape[0] == 1
        x = np.tile(x, reps=(u.shape[0], 1, 1, 1))
      else:
        assert x.shape[0] == u.shape[0]

      # Define temporal coordinates
      if self.metadata.domain_t is not None:
        t = np.linspace(*self.metadata.domain_t, u.shape[1], endpoint=True)
        t = t.reshape(1, -1, 1, 1)
        # Repeat along sample trajectory
        t = np.tile(t, reps=(u.shape[0], 1, 1, 1))
      else:
        t = None

    else:
      raise ValueError

    # Only Keep the desired variables
    if self.idx_vars is not None:
      u = u[..., self.idx_vars]

    # Cut the time axis
    if self.time_dependent and self.time_cutoff_idx:
      u = u[:, :self.time_cutoff_idx]
      if c is not None: c = c[:, :self.time_cutoff_idx]
      t = t[:, :self.time_cutoff_idx]
      x = x[:, :self.time_cutoff_idx]

    # Downsample in the time axis
    if self.time_dependent and self.time_downsample_factor > 1:
      u = u[:, ::self.time_downsample_factor]
      if c is not None: c = c[:, ::self.time_downsample_factor]
      if t is not None: t = t[:, ::self.time_downsample_factor]
      x = x[:, ::self.time_downsample_factor]

    # Downsample the space coordinates randomly
    if self.space_downsample_factor > 1:
      permutation = jax.random.permutation(self.key, u.shape[2])
      # NOTE: Same discretization for all snapshots
      u = u[:, :, permutation]
      c = c[:, :, permutation] if (c is not None) else None
      x = x[:, :, permutation]

      size = int(u.shape[2] / (self.space_downsample_factor ** 2))
      u = u[:, :, :size]
      c = c[:, :, :size] if (c is not None) else None
      x = x[:, :, :size]

    if self.concatenate_coeffs and (c is not None):
      u = np.concatenate([u, c], axis=-1)
      c = None

    batch = Batch(u=u, c=c, x=x, t=t, g=g)

    return batch

  def _fetch_mode(self, idx: Union[int, Sequence], mode: str, get_graphs: bool = True):
    """Fetches a sample from a specific part of the dataset."""

    # Check inputs
    if isinstance(idx, int):
      idx = [idx]
    # Set mode index
    assert all([i < len(self.idx_modes[mode]) for i in idx])
    _idx = self.idx_modes[mode][np.array(idx)]

    return self._fetch(_idx, get_graphs=get_graphs)

  def train(self, idx: Union[int, Sequence]):
    """Fetches a sample from the training samples."""

    return self._fetch_mode(idx, mode='train')

  def valid(self, idx: Union[int, Sequence]):
    """Fetches a sample from the validation samples."""

    return self._fetch_mode(idx, mode='valid')

  def test(self, idx: Union[int, Sequence]):
    """Fetches a sample from the test samples."""

    return self._fetch_mode(idx, mode='test')

  def batches(self, mode: str, batch_size: int, get_graphs: bool = True, key: PRNGKey = None):
    """Fetches batches of samples from a specific part of the dataset."""

    assert batch_size > 0
    assert batch_size <= self.nums[mode]

    if key is not None:
      _idx_mode_permuted = jax.random.permutation(key, np.arange(self.nums[mode]))
    else:
      _idx_mode_permuted = jnp.arange(self.nums[mode])

    len_dividable = self.nums[mode] - (self.nums[mode] % batch_size)
    for idx in np.split(_idx_mode_permuted[:len_dividable], len_dividable // batch_size):
      batch = self._fetch_mode(idx, mode, get_graphs=get_graphs)
      yield batch

    if (self.nums[mode] % batch_size):
      idx = _idx_mode_permuted[len_dividable:]
      batch = self._fetch_mode(idx, mode, get_graphs=get_graphs)
      yield batch

  def __len__(self):
    return self.length
