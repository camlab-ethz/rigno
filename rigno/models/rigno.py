from typing import Tuple, Union, NamedTuple

import flax.typing
import jax.numpy as jnp
import jax.random
import numpy as np
import jraph
from flax import linen as nn
from scipy.spatial import Delaunay

from rigno.graph.entities import (
    TypedGraph, EdgeSet, EdgeSetKey,
    EdgesIndices, NodeSet, Context)
from rigno.models.graphnet import DeepTypedGraphNet
from rigno.models.operator import AbstractOperator, Inputs
from rigno.utils import Array, shuffle_arrays


class RegionInteractionGraphSet(NamedTuple):
  p2r: TypedGraph
  r2r: TypedGraph
  r2p: TypedGraph

  def __len__(self) -> int:
    return self.p2r.nodes['pnodes'].n_node.shape[0]

class RegionInteractionGraphMetadata(NamedTuple):
  """Light-weight class for storing graph metadata."""

  x_pnodes_inp: Array
  x_pnodes_out: Array
  x_rnodes: Array
  r_rnodes: Array
  p2r_edge_indices: Array
  r2r_edge_indices: Array
  r2r_edge_domains: Array
  r2p_edge_indices: Array

  def __len__(self) -> int:
    return self.x_pnodes_inp.shape[0]

class RegionInteractionGraphBuilder:

  def __init__(self,
    periodic: bool,
    rmesh_levels: int,
    subsample_factor: float,
    overlap_factor_p2r: float,
    overlap_factor_r2p: float,
    node_coordinate_freqs: int
  ):

    # Set attributes
    self.periodic = periodic
    self.overlap_factor_p2r = overlap_factor_p2r
    self.overlap_factor_r2p = overlap_factor_r2p
    self.node_coordinate_freqs = node_coordinate_freqs
    self.rmesh_levels = rmesh_levels
    self.subsample_factor = subsample_factor

    # Domain shifts for periodic BC
    self._domain_shifts = jnp.concatenate([
      jnp.array([[0., 0.]]),  # C
      jnp.array([[-2, 0.]]),  # W
      jnp.array([[-2, +2]]),  # NW
      jnp.array([[0., +2]]),  # N
      jnp.array([[+2, +2]]),  # NE
      jnp.array([[+2, 0.]]),  # E
      jnp.array([[+2, -2]]),  # SE
      jnp.array([[0., -2]]),  # S
      jnp.array([[-2, -2]]),  # SW
    ], axis=0)

  def _compute_minimum_support_radius(self, x: Array) -> Array:
      # NOTE: This function is not jittable because of the Delaunay triangulation

      if self.periodic:
        x_extended = (x[None, :, :] + self._domain_shifts[:, None, :]).reshape(-1, 2)
        tri = Delaunay(points=x_extended)
      else:
        tri = Delaunay(points=x)

      medians = _compute_triangulation_medians(tri)
      radii = np.zeros(shape=(x.shape[0],))
      mask = tri.simplices < x.shape[0] # [N, 3]
      values = medians[mask]
      indices = tri.simplices[mask]
      sorted_idx = np.argsort(indices)
      sorted_indices = indices[sorted_idx]
      sorted_values = values[sorted_idx]
      unique_indices, idx_start = np.unique(sorted_indices, return_index=True)
      radii[unique_indices] = np.maximum.reduceat(sorted_values, idx_start)

      return radii

  def _get_supported_pnodes_by_rnodes(self,
    centers: Array,
    points: Array,
    radii: Array,
    ord_distance: int = 2,
  ) -> Array:
    """ord_distance can be 1, 2, or np.inf"""

    # Replace large radii
    # NOTE: Makeshift solution for peculiar geometries
    # TODO: Instead, remove out-of-domain mesh edges in order to avoid large radiuses
    radii = np.where(radii < .5, radii, .2)

    # Get relative coordinates
    rel = points[:, None] - centers
    # Mirror relative positions because of periodic boudnary conditions
    if self.periodic:
      rel = jnp.where(rel >= 1., (rel - 2.), rel)
      rel = jnp.where(rel < -1., (rel + 2.), rel)

    # Compute distance
    # NOTE: Order of the norm determines the shape of the sub-regions
    distance = jnp.linalg.norm(rel, ord=ord_distance, axis=-1)

    # Get indices
    # -> [idx_point, idx_center]
    idx_nodes = jnp.stack(jnp.where(distance <= radii), axis=-1)

    return idx_nodes

  def _get_r2r_edges(self, x_rmesh: Array) -> Tuple[Array, Array]:
    """Constructrs the processor graph (rmesh to rmesh)"""

    # Define edges and their corresponding -extended- domain
    edges = []
    domains = []
    for level in range(self.rmesh_levels):
      # Sub-sample the rmesh
      _rmesh_size = int(x_rmesh.shape[0] / (self.subsample_factor ** level))
      if _rmesh_size < 4:
        continue
      _x_rmesh = x_rmesh[:_rmesh_size]
      # Construct a triangulation
      if self.periodic:
        # Repeat the rmesh in periodic directions
        _x_rmesh_extended = (_x_rmesh[None, :, :] + self._domain_shifts[:, None, :]).reshape(-1, 2)
        tri = Delaunay(points=_x_rmesh_extended)
      else:
        tri = Delaunay(points=_x_rmesh)
      # Get the relevant edges
      _extended_edges = _get_edges_from_triangulation(tri)
      domains_level = _extended_edges // _rmesh_size
      edges_level = _extended_edges % _rmesh_size
      idx_relevant_edges = np.any(domains_level == 0, axis=1) if self.periodic else np.all(domains_level == 0, axis=1)
      edges_level = edges_level[idx_relevant_edges]
      domains_level = domains_level[idx_relevant_edges]
      edges.append(edges_level)
      domains.append(domains_level)

    # Remove repeated edges
    edges = jnp.concatenate(edges)
    domains = jnp.concatenate(domains)
    _, unique_idx = jnp.unique(edges, axis=0, return_index=True)
    edges = edges[unique_idx]
    domains = domains[unique_idx]

    return edges, domains

  # TODO: Build multiple graphs at the same time in a vectorial way
  def build_metadata(self, x_inp: Array, x_out: Array, domain: Array, rmesh_correction_dsf: int = 1, key: Union[flax.typing.PRNGKey, None] = None) -> RegionInteractionGraphMetadata:

    # Normalize coordinates in [-1, +1)
    x_inp = 2 * (x_inp - domain[0]) / (domain[1] - domain[0]) - 1
    x_out = 2 * (x_out - domain[0]) / (domain[1] - domain[0]) - 1

    # Randomly sub-sample pmesh to get rmesh
    if key is None: key = jax.random.PRNGKey(0)
    if self.periodic:
      x_rnodes = _subsample_pointset(key=key, x=x_inp, factor=self.subsample_factor)
    else:
      # TODO: Keep boundary nodes for non-periodic BC
      x_rnodes = _subsample_pointset(key=key, x=x_inp, factor=self.subsample_factor)
      # NOTE: With the below implementation the number of x_rnodes might vary
      # idx_bound = np.where((x_inp[:, 0] == -1) | (x_inp[:, 0] == +1) | (x_inp[:, 1] == -1) | (x_inp[:, 1] == +1))
      # x_boundary = x_inp[idx_bound]
      # x_internal = np.delete(x_inp, idx_bound, axis=0)
      # x_rmesh_internal = _subsample_pointset(key=key, x=x_internal, factor=self.subsample_factor)
      # x_rnodes, = shuffle_arrays(key=key, arrays=(jnp.concatenate([x_boundary, x_rmesh_internal]),))

    # Downsample or upsample the rmesh
    if rmesh_correction_dsf > 1:
      x_rnodes = _subsample_pointset(key=key, x=x_rnodes, factor=rmesh_correction_dsf)
    elif rmesh_correction_dsf < 1:
      x_rnodes = _upsample_pointset(key=key, x=x_rnodes, factor=(1 / rmesh_correction_dsf))

    # Compute minimum support radius of each rmesh node
    r_rnodes = self._compute_minimum_support_radius(x_rnodes)

    # Get edge indices
    p2r_edge_indices = self._get_supported_pnodes_by_rnodes(
      centers=x_rnodes,
      points=x_inp,
      radii=jnp.clip(self.overlap_factor_p2r * r_rnodes, a_min=0, a_max=r_rnodes.max()),
    )
    r2r_edge_indices, r2r_edge_domains = self._get_r2r_edges(x_rnodes)
    r2p_edge_indices = self._get_supported_pnodes_by_rnodes(
      centers=x_rnodes,
      points=x_inp,
      radii=jnp.clip(self.overlap_factor_r2p * r_rnodes, a_min=0, a_max=r_rnodes.max()),
    )
    r2p_edge_indices = jnp.flip(r2p_edge_indices, axis=-1)

    # Add dummy nodes and edges
    p2r_edge_indices = jnp.concatenate([p2r_edge_indices, jnp.array([[x_inp.shape[0], x_rnodes.shape[0]]])], axis=0)
    r2r_edge_indices = jnp.concatenate([r2r_edge_indices, jnp.array([[x_rnodes.shape[0], x_rnodes.shape[0]]])], axis=0)
    r2r_edge_domains = jnp.concatenate([r2r_edge_domains, jnp.array([[0, 0]])], axis=0)
    r2p_edge_indices = jnp.concatenate([r2p_edge_indices, jnp.array([[x_rnodes.shape[0], x_out.shape[0]]])], axis=0)
    x_inp = jnp.concatenate([x_inp, jnp.zeros(shape=(1, x_inp.shape[-1]))], axis=0)
    x_out = jnp.concatenate([x_out, jnp.zeros(shape=(1, x_out.shape[-1]))], axis=0)
    x_rnodes = jnp.concatenate([x_rnodes, jnp.zeros(shape=(1, x_rnodes.shape[-1]))], axis=0)
    r_rnodes = jnp.concatenate([r_rnodes, jnp.zeros(shape=(1,))], axis=0)

    # Convert dtypes to save memory
    r2r_edge_domains = r2r_edge_domains.astype(jnp.uint8)
    if (max(x_inp.shape[0], x_out.shape[0]) < jnp.iinfo(jnp.uint16).max):
      p2r_edge_indices=p2r_edge_indices.astype(jnp.uint16)
      r2r_edge_indices=r2r_edge_indices.astype(jnp.uint16)
      r2p_edge_indices=r2p_edge_indices.astype(jnp.uint16)
    # Ommit storing duplicated edge indices
    if self.overlap_factor_p2r == self.overlap_factor_r2p:
      # NOTE: it will be the inverse of p2r edges
      r2p_edge_indices = None

    # Store the graph data
    graph_metadata = RegionInteractionGraphMetadata(
      x_pnodes_inp=jnp.expand_dims(x_inp, axis=0),
      x_pnodes_out=jnp.expand_dims(x_out, axis=0),
      x_rnodes=jnp.expand_dims(x_rnodes, axis=0),
      r_rnodes=jnp.expand_dims(r_rnodes, axis=0),
      p2r_edge_indices=jnp.expand_dims(p2r_edge_indices, axis=0),
      r2r_edge_indices=jnp.expand_dims(r2r_edge_indices, axis=0),
      r2r_edge_domains=jnp.expand_dims(r2r_edge_domains, axis=0),
      r2p_edge_indices=(jnp.expand_dims(r2p_edge_indices, axis=0) if (r2p_edge_indices is not None) else None),
    )

    return graph_metadata

  def _init_structural_features(self,
    x_sen: Array,
    x_rec: Array,
    idx_sen: Array,
    idx_rec: Array,
    max_edge_length: float,
    feats_sen: Array = None,
    feats_rec: Array = None,
    shift: bool = False,
    domain_sen: Array = None,
    domain_rec: Array = None,
  ) -> Tuple[EdgeSet, NodeSet, NodeSet]:

    # Get number of nodes and the edges
    batch_size = x_sen.shape[0]
    num_sen = x_sen.shape[1]
    num_rec = x_rec.shape[1]
    assert idx_sen.shape[1] == idx_rec.shape[1]
    num_edg = idx_sen.shape[1]

    # Process coordinates
    phi_sen = jnp.pi * (x_sen + 1)  # [0, 2pi]
    phi_rec = jnp.pi * (x_rec + 1)  # [0, 2pi]

    # Define node features
    # NOTE: Sinusoidal features don't need normalization
    if self.periodic:
      k = jnp.arange(self.node_coordinate_freqs)
      phi_sen_sin = jax.vmap(fun=(lambda _v, _k: jnp.sin(_v * (_k+1))), in_axes=(None, 0), out_axes=-1)(phi_sen, k)
      phi_sen_cos = jax.vmap(fun=(lambda _v, _k: jnp.cos(_v * (_k+1))), in_axes=(None, 0), out_axes=-1)(phi_sen, k)
      sender_node_feats = jnp.concatenate(
        arrays=[
          phi_sen_sin.reshape(*phi_sen_sin.shape[:-2], -1),
          phi_sen_cos.reshape(*phi_sen_cos.shape[:-2], -1)
        ], axis=-1)
      phi_rec_sin = jax.vmap(fun=(lambda _v, _k: jnp.sin(_v * (_k+1))), in_axes=(None, 0), out_axes=-1)(phi_rec, k)
      phi_rec_cos = jax.vmap(fun=(lambda _v, _k: jnp.cos(_v * (_k+1))), in_axes=(None, 0), out_axes=-1)(phi_rec, k)
      receiver_node_feats = jnp.concatenate(
        arrays=[
          phi_rec_sin.reshape(*phi_rec_sin.shape[:-2], -1),
          phi_rec_cos.reshape(*phi_rec_cos.shape[:-2], -1)
        ], axis=-1)
    else:
      sender_node_feats = jnp.concatenate([x_sen], axis=-1)
      receiver_node_feats = jnp.concatenate([x_rec], axis=-1)
    # Concatenate with forced features
    if feats_sen is not None:
      sender_node_feats = jnp.concatenate([sender_node_feats, feats_sen], axis=-1)
    if feats_rec is not None:
      receiver_node_feats = jnp.concatenate([receiver_node_feats, feats_rec], axis=-1)

    # Build node sets
    sender_node_set = NodeSet(
      n_node=jnp.tile(jnp.array([num_sen]), reps=(batch_size, 1)),
      features=sender_node_feats,
    )
    receiver_node_set = NodeSet(
      n_node=jnp.tile(jnp.array([num_rec]), reps=(batch_size, 1)),
      features=receiver_node_feats,
    )

    # Define edge features
    batched_index = jax.vmap(lambda f, idx: f[idx])
    batched_index_single = jax.vmap(lambda f, idx: f[idx], in_axes=(None, 0))
    z_ij = batched_index(x_sen, idx_sen) - batched_index(x_rec, idx_rec)
    if self.periodic:
      # NOTE: For p2r and r2p, mirror the large relative coordinates
      # MODIFY: Unify the mirroring with the below method in r2r
      if not shift:
        z_ij = jnp.where(z_ij < -1.0, z_ij + 2, z_ij)
        z_ij = jnp.where(z_ij >= 1.0, z_ij - 2, z_ij)
      # NOTE: For the r2r multi-mesh, use extended domain indices and shifts
      else:
        z_ij = (
          (batched_index(x_sen, idx_sen) + batched_index_single(self._domain_shifts, domain_sen))
          - (batched_index(x_rec, idx_rec) + batched_index_single(self._domain_shifts, domain_rec))
        )
    d_ij = jnp.linalg.norm(z_ij, axis=-1, keepdims=True)
    # Normalize and concatenate edge features
    z_ij = z_ij / max_edge_length
    d_ij = d_ij / max_edge_length
    edge_feats = jnp.concatenate([z_ij, d_ij], axis=-1)

    # Build edge set
    edge_set = EdgeSet(
      n_edge=jnp.tile(jnp.array([num_edg]), reps=(batch_size, 1)),
      indices=EdgesIndices(
        senders=idx_sen,
        receivers=idx_rec,
      ),
      features=edge_feats,
    )

    return edge_set, sender_node_set, receiver_node_set

  def _build_p2r_graph(self, x_pnodes: Array, x_rnodes: Array, idx_edges: Array, r_rmesh: Array) -> TypedGraph:
    """Constructs the encoder graph (pmesh to rmesh)"""

    # Get the initial features
    edge_set, pmesh_node_set, rmesh_node_set = self._init_structural_features(
      x_sen=x_pnodes,
      x_rec=x_rnodes,
      idx_sen=idx_edges[..., 0],
      idx_rec=idx_edges[..., 1],
      max_edge_length=(2. * jnp.sqrt(x_rnodes.shape[-1])),
      feats_rec=jnp.expand_dims(self.overlap_factor_p2r * r_rmesh, axis=-1),
    )

    # Construct the graph
    graph = TypedGraph(
      context=Context(n_graph=jnp.tile(jnp.array([1]), reps=(x_rnodes.shape[0], 1)), features=()),
      nodes={'pnodes': pmesh_node_set, 'rnodes': rmesh_node_set},
      edges={EdgeSetKey('p2r', ('pnodes', 'rnodes')): edge_set},
    )

    return graph

  def _build_r2r_graph(self, x_rnodes: Array, idx_edges: Array, idx_domains: Array, r_rmesh: Array) -> TypedGraph:
    """Constructs the processor graph (rmesh to rmesh)"""

    # Set the initial features
    edge_set, rmesh_node_set, _ = self._init_structural_features(
      x_sen=x_rnodes,
      x_rec=x_rnodes,
      idx_sen=idx_edges[..., 0],
      idx_rec=idx_edges[..., 1],
      max_edge_length=(2. * jnp.sqrt(x_rnodes.shape[-1])),
      feats_sen=jnp.expand_dims(self.overlap_factor_p2r * r_rmesh, axis=-1),
      feats_rec=jnp.expand_dims(self.overlap_factor_r2p * r_rmesh, axis=-1),
      shift=True,
      domain_sen=idx_domains[..., 0],
      domain_rec=idx_domains[..., 1],
    )

    # Construct the graph
    graph = TypedGraph(
      context=Context(n_graph=jnp.tile(jnp.array([1]), reps=(x_rnodes.shape[0], 1)), features=()),
      nodes={'rnodes': rmesh_node_set},
      edges={EdgeSetKey('r2r', ('rnodes', 'rnodes')): edge_set},
    )

    return graph

  def _build_r2p_graph(self, x_pnodes: Array, x_rnodes: Array, idx_edges: Array, r_rmesh: Array) -> TypedGraph:
    """Constructs the decoder graph (rmesh to pmesh)"""

    # Get the initial features
    edge_set, rmesh_node_set, pmesh_node_set = self._init_structural_features(
      x_sen=x_rnodes,
      x_rec=x_pnodes,
      idx_sen=idx_edges[..., 0],
      idx_rec=idx_edges[..., 1],
      max_edge_length=(2. * jnp.sqrt(x_rnodes.shape[-1])),
      feats_sen=jnp.expand_dims(self.overlap_factor_r2p * r_rmesh, axis=-1),
    )

    # Construct the graph
    graph = TypedGraph(
      context=Context(n_graph=jnp.tile(jnp.array([1]), reps=(x_rnodes.shape[0], 1)), features=()),
      nodes={'pnodes': pmesh_node_set, 'rnodes': rmesh_node_set},
      edges={EdgeSetKey('r2p', ('rnodes', 'pnodes')): edge_set},
    )

    return graph

  def build_graphs(self, metadata: RegionInteractionGraphMetadata) -> RegionInteractionGraphSet:

    # Unwrap the attributes
    x_pnodes_inp = metadata.x_pnodes_inp
    x_pnodes_out = metadata.x_pnodes_out
    x_rnodes = metadata.x_rnodes
    r_rnodes = metadata.r_rnodes
    p2r_edge_indices = metadata.p2r_edge_indices
    r2r_edge_indices = metadata.r2r_edge_indices
    r2r_edge_domains = metadata.r2r_edge_domains
    r2p_edge_indices = metadata.r2p_edge_indices
    # Flip p2r indices if r2p is None
    if r2p_edge_indices is None:
      r2p_edge_indices = jnp.flip(metadata.p2r_edge_indices, axis=-1)

    # Build the graphs
    graphs = RegionInteractionGraphSet(
      p2r=self._build_p2r_graph(x_pnodes_inp, x_rnodes, p2r_edge_indices, r_rnodes),
      r2r=self._build_r2r_graph(x_rnodes, r2r_edge_indices, r2r_edge_domains, r_rnodes),
      r2p=self._build_r2p_graph(x_pnodes_out, x_rnodes, r2p_edge_indices, r_rnodes),
    )

    return graphs

class Encoder(nn.Module):
  node_latent_size: int
  edge_latent_size: int
  mlp_hidden_layers: int = 1
  use_layer_norm: bool = True
  conditioned_normalization: bool = True
  cond_norm_hidden_size: bool = True
  p_edge_masking: float = .0

  def setup(self):
    self.gnn = DeepTypedGraphNet(
      embed_nodes=True,  # Embed raw features of all nodes
      embed_edges=True,  # Embed raw features of the edges
      edge_latent_size=dict(p2r=self.edge_latent_size),
      node_latent_size=dict(rnodes=self.node_latent_size, pnodes=self.node_latent_size),
      mlp_num_hidden_layers=self.mlp_hidden_layers,
      num_message_passing_steps=1,
      use_layer_norm=self.use_layer_norm,
      conditioned_normalization=self.conditioned_normalization,
      cond_norm_hidden_size=self.cond_norm_hidden_size,
      include_sent_messages_in_node_update=False,
      activation='swish',
      f32_aggregation=True,
      aggregate_edges_for_nodes_fn=jraph.segment_mean,
    )

  def __call__(self,
    graph: TypedGraph,
    pnode_features: Array,
    tau: Union[None, float],
    key: Union[flax.typing.PRNGKey, None] = None,
  ) -> tuple[Array, Array]:
    """Runs the p2r GNN, extracting latent physical and regional nodes."""

    # Get batch size
    batch_size = pnode_features.shape[0]

    # Concatenate node structural features with input features
    pnodes = graph.nodes['pnodes']
    rnodes = graph.nodes['rnodes']
    new_pnodes = pnodes._replace(
      features=jnp.concatenate([pnode_features, pnodes.features], axis=-1)
    )
    # CHECK: Is this necessary?
    # To make sure capacity of the embedded is identical for the physical nodes and
    # the regional nodes, we also append some dummy zero input features for the
    # regional nodes.
    dummy_rnode_features = jnp.zeros(
        rnodes.features.shape[:2] + (pnode_features.shape[-1],),
        dtype=pnode_features.dtype)
    new_rnodes = rnodes._replace(
      features=jnp.concatenate([dummy_rnode_features, rnodes.features], axis=-1)
    )

    # Get edges
    p2r_edges_key = graph.edge_key_by_name('p2r')
    edges = graph.edges[p2r_edges_key]
    # Drop out edges randomly with the given probability
    if key is not None:
      n_edges_after = int((1 - self.p_edge_masking) * edges.features.shape[1])
      [new_edge_features, new_edge_senders, new_edge_receivers] = shuffle_arrays(
        key=key, arrays=[edges.features, edges.indices.senders, edges.indices.receivers], axis=1)
      new_edge_features = new_edge_features[:, :n_edges_after]
      new_edge_senders = new_edge_senders[:, :n_edges_after]
      new_edge_receivers = new_edge_receivers[:, :n_edges_after]
    else:
      n_edges_after = edges.features.shape[1]
      new_edge_features = edges.features
      new_edge_senders = edges.indices.senders
      new_edge_receivers = edges.indices.receivers
    # Change edge feature dtype
    new_edge_features = new_edge_features.astype(dummy_rnode_features.dtype)
    # Build new edge set
    new_edges = EdgeSet(
      n_edge=jnp.tile(jnp.array([n_edges_after]), reps=(batch_size, 1)),
      indices=EdgesIndices(
        senders=new_edge_senders,
        receivers=new_edge_receivers,
      ),
      features=new_edge_features,
    )

    input_graph = graph._replace(
      edges={p2r_edges_key: new_edges},
      nodes={
        'pnodes': new_pnodes,
        'rnodes': new_rnodes
      })

    # Run the GNN
    p2r_out = self.gnn(input_graph, condition=tau)
    latent_rnodes = p2r_out.nodes['rnodes'].features
    latent_pnodes = p2r_out.nodes['pnodes'].features

    return latent_rnodes, latent_pnodes

class Processor(nn.Module):
  steps: int
  node_latent_size: int
  edge_latent_size: int
  mlp_hidden_layers: int = 1
  use_layer_norm: bool = True
  conditioned_normalization: bool = True
  cond_norm_hidden_size: bool = True
  p_edge_masking: float = .0

  def setup(self):
    self.gnn = DeepTypedGraphNet(
      embed_nodes=False,  # Node features already embdded by previous layers
      embed_edges=True,  # Embed raw features of the edges
      edge_latent_size=dict(r2r=self.edge_latent_size),
      node_latent_size=dict(rnodes=self.node_latent_size),
      mlp_num_hidden_layers=self.mlp_hidden_layers,
      num_message_passing_steps=self.steps,
      use_layer_norm=True,
      conditioned_normalization=self.conditioned_normalization,
      cond_norm_hidden_size=self.cond_norm_hidden_size,
      include_sent_messages_in_node_update=False,
      activation='swish',
      f32_aggregation=False,
      # NOTE: segment_mean because number of edges is not balanced
      aggregate_edges_for_nodes_fn=jraph.segment_mean,
    )

  def __call__(self,
    graph: TypedGraph,
    rnode_features: Array,
    tau: Union[None, float],
    key: Union[flax.typing.PRNGKey, None] = None,
  ) -> Array:
    """Runs the r2r GNN, extracting updated latent regional nodes."""

    # Get batch size
    batch_size = rnode_features.shape[0]

    # Replace the node features
    # NOTE: We don't need to add the structural node features, because these are
    # already part of  the latent state, via the original p2r gnn.
    rnodes = graph.nodes['rnodes']
    new_rnodes = rnodes._replace(features=rnode_features)

    # Get edges
    r2r_edges_key = graph.edge_key_by_name('r2r')
    # NOTE: We are assuming here that the r2r gnn uses a single set of edge keys
    # named 'r2r' for the edges and that it uses a single set of nodes named 'rnodes'
    msg = ('The setup currently requires to only have one kind of edge in the mesh GNN.')
    assert len(graph.edges) == 1, msg
    edges = graph.edges[r2r_edges_key]
    # Drop out edges randomly with the given probability
    # NOTE: We need the structural edge features, because it is the first
    # time we are seeing this particular set of edges.
    if key is not None:
      n_edges_after = int((1 - self.p_edge_masking) * edges.features.shape[1])
      [new_edge_features, new_edge_senders, new_edge_receivers] = shuffle_arrays(
        key=key, arrays=[edges.features, edges.indices.senders, edges.indices.receivers], axis=1)
      new_edge_features = new_edge_features[:, :n_edges_after]
      new_edge_senders = new_edge_senders[:, :n_edges_after]
      new_edge_receivers = new_edge_receivers[:, :n_edges_after]
    else:
      n_edges_after = edges.features.shape[1]
      new_edge_features = edges.features
      new_edge_senders = edges.indices.senders
      new_edge_receivers = edges.indices.receivers
    # Change edge feature dtype
    new_edge_features = new_edge_features.astype(rnode_features.dtype)
    # Build new edge set
    new_edges = EdgeSet(
      n_edge=jnp.tile(jnp.array([n_edges_after]), reps=(batch_size, 1)),
      indices=EdgesIndices(
        senders=new_edge_senders,
        receivers=new_edge_receivers,
      ),
      features=new_edge_features,
    )

    # Build the graph
    input_graph = graph._replace(
      edges={r2r_edges_key: new_edges},
      nodes={'rnodes': new_rnodes},
    )

    # Run the GNN
    output_graph = self.gnn(input_graph, condition=tau)
    output_rnodes = output_graph.nodes['rnodes'].features

    return output_rnodes

class Decoder(nn.Module):
  variable_mesh: bool
  num_outputs: int
  node_latent_size: int
  edge_latent_size: int
  mlp_hidden_layers: int = 1
  use_layer_norm: bool = True
  conditioned_normalization: bool = True
  cond_norm_hidden_size: bool = True
  p_edge_masking: float = .0

  def setup(self):
    self.gnn = DeepTypedGraphNet(
    # NOTE: with variable mesh, the output pnode features must be embedded
    embed_nodes=(dict(pnodes=True) if self.variable_mesh else False),
    embed_edges=True,  # Embed raw features of the edges
    # Require a specific node dimensionaly for the physical node outputs
    # NOTE: This triggers the independent mapping for pnodes
    node_output_size=dict(pnodes=self.num_outputs),
    edge_latent_size=dict(r2p=self.edge_latent_size),
    node_latent_size=dict(rnodes=self.node_latent_size, pnodes=self.node_latent_size),
    mlp_num_hidden_layers=self.mlp_hidden_layers,
    num_message_passing_steps=1,
    use_layer_norm=True,
    conditioned_normalization=self.conditioned_normalization,
    cond_norm_hidden_size=self.cond_norm_hidden_size,
    include_sent_messages_in_node_update=False,
    activation='swish',
    f32_aggregation=False,
    # NOTE: segment_mean because number of edges is not balanced
    aggregate_edges_for_nodes_fn=jraph.segment_mean,
  )

  def __call__(self,
    graph: TypedGraph,
    rnode_features: Array,
    pnode_features: Array,
    tau: Union[None, float],
    key: Union[flax.typing.PRNGKey, None] = None,
  ) -> Array:
    """Runs the r2p GNN, extracting the output physical nodes."""

    # Get batch size
    batch_size = rnode_features.shape[0]

    # NOTE: We don't need to add the structural node features, because these are
    # already part of the latent state, via the original p2r gnn.
    rnodes = graph.nodes['rnodes']
    pnodes = graph.nodes['pnodes']
    new_rnodes = rnodes._replace(features=rnode_features)
    if self.variable_mesh:
      # NOTE: We can't use latent pnodes of the input mesh for the output mesh
      # TRY: Make sure that this does not harm the performance with fixed mesh
      # If it works, change the architecture, flowcharts, etc.
      new_pnodes = pnodes._replace(features=pnodes.features)
    else:
      new_pnodes = pnodes._replace(features=pnode_features)

    # Get edges
    r2p_edges_key = graph.edge_key_by_name('r2p')
    edges = graph.edges[r2p_edges_key]
    # Drop out edges randomly with the given probability
    if key is not None:
      n_edges_after = int((1 - self.p_edge_masking) * edges.features.shape[1])
      [new_edge_features, new_edge_senders, new_edge_receivers] = shuffle_arrays(
        key=key, arrays=[edges.features, edges.indices.senders, edges.indices.receivers], axis=1)
      new_edge_features = new_edge_features[:, :n_edges_after]
      new_edge_senders = new_edge_senders[:, :n_edges_after]
      new_edge_receivers = new_edge_receivers[:, :n_edges_after]
    else:
      n_edges_after = edges.features.shape[1]
      new_edge_features = edges.features
      new_edge_senders = edges.indices.senders
      new_edge_receivers = edges.indices.receivers
    # Change edge feature dtype
    new_edge_features = new_edge_features.astype(pnode_features.dtype)
    # Build new edge set
    new_edges = EdgeSet(
      n_edge=jnp.tile(jnp.array([n_edges_after]), reps=(batch_size, 1)),
      indices=EdgesIndices(
        senders=new_edge_senders,
        receivers=new_edge_receivers,
      ),
      features=new_edge_features,
    )

    # Build the new graph
    input_graph = graph._replace(
      edges={r2p_edges_key: new_edges},
      nodes={
        'rnodes': new_rnodes,
        'pnodes': new_pnodes
      })

    # Run the GNN
    output_graph = self.gnn(input_graph, condition=tau)
    output_pnodes = output_graph.nodes['pnodes'].features

    return output_pnodes

class RIGNO(AbstractOperator):
  """TODO: Add docstrings"""

  num_outputs: int
  processor_steps: int = 18
  node_latent_size: int = 128
  edge_latent_size: int = 128
  mlp_hidden_layers: int = 1
  concatenate_t: bool = True
  concatenate_tau: bool = True
  conditioned_normalization: bool = True
  cond_norm_hidden_size: int = 16
  p_edge_masking: int = 0.5

  def _check_coordinates(self, x: Array) -> None:
    assert x is not None
    assert x.ndim == 2
    assert x.shape[1] <= 3
    assert x.min() >= -1
    assert x.max() <= +1

  def _check_function(self, u: Array, x: Array) -> None:
    assert u is not None
    assert u.ndim == 4
    assert u.shape[1] == 1
    assert u.shape[2] == x.shape[2], f'u: {u.shape}, x: {x.shape}'

  def setup(self):
    # NOTE: There are a few architectural considerations for variable mesh
    # NOTE: variable_mesh=True means that the input and the output mesh can be different
    # NOTE: Check usages of this attribute
    self.variable_mesh = False

    self.encoder = Encoder(
      edge_latent_size=self.edge_latent_size,
      node_latent_size=self.node_latent_size,
      mlp_hidden_layers=self.mlp_hidden_layers,
      conditioned_normalization=self.conditioned_normalization,
      cond_norm_hidden_size=self.cond_norm_hidden_size,
      p_edge_masking=self.p_edge_masking,
      name='encoder',
    )

    self.processor = Processor(
      steps=self.processor_steps,
      edge_latent_size=self.edge_latent_size,
      node_latent_size=self.node_latent_size,
      mlp_hidden_layers=self.mlp_hidden_layers,
      conditioned_normalization=self.conditioned_normalization,
      cond_norm_hidden_size=self.cond_norm_hidden_size,
      p_edge_masking=self.p_edge_masking,
      name='processor',
    )

    self.decoder = Decoder(
      variable_mesh=self.variable_mesh,
      num_outputs=self.num_outputs,
      edge_latent_size=self.edge_latent_size,
      node_latent_size=self.node_latent_size,
      mlp_hidden_layers=self.mlp_hidden_layers,
      conditioned_normalization=self.conditioned_normalization,
      cond_norm_hidden_size=self.cond_norm_hidden_size,
      p_edge_masking=self.p_edge_masking,
      name='decoder',
    )

  @staticmethod
  def _prepare_features(feats: Array) -> Array:
    # Expand time axis
    feats = jnp.expand_dims(feats, axis=1)
    return feats

  def _encode_process_decode(self,
    graphs: RegionInteractionGraphSet,
    pnode_features: Array,
    tau: Union[None, float],
    key: flax.typing.PRNGKey = None,
  ) -> Array:

    # Add dummy node features
    dummy_pnode_features = jnp.zeros(shape=(pnode_features.shape[0], 1, pnode_features.shape[2]))
    pnode_features = jnp.concatenate([pnode_features, dummy_pnode_features], axis=1)

    # Transfer data for the physical mesh to the regional mesh
    # -> [batch_size, num_nodes, latent_size]
    subkey, key = jax.random.split(key) if (key is not None) else (None, None)
    (latent_rnodes, latent_pnodes) = self.encoder(graphs.p2r, pnode_features, tau, key=subkey)
    self.sow(
      col='intermediates', name='pnodes_encoded',
      value=self._prepare_features(latent_pnodes[:, :-1])
    )
    self.sow(
      col='intermediates', name='rnodes_encoded',
      value=self._prepare_features(latent_rnodes[:, :-1])
    )

    # Run message-passing in the regional mesh
    # -> [batch_size, num_rnodes, latent_size]
    subkey, key = jax.random.split(key) if (key is not None) else (None, None)
    updated_latent_rnodes = self.processor(graphs.r2r, latent_rnodes, tau, key=subkey)
    self.sow(
      col='intermediates', name='rnodes_processed',
      value=self._prepare_features(updated_latent_rnodes[:, :-1])
    )

    # Transfer data from the regional mesh to the physical mesh
    # -> [batch_size, num_pnodes_out, latent_size]
    subkey, key = jax.random.split(key) if (key is not None) else (None, None)
    output_pnodes = self.decoder(graphs.r2p, updated_latent_rnodes, latent_pnodes, tau, key=subkey)
    self.sow(
      col='intermediates', name='pnodes_decoded',
      value=self._prepare_features(output_pnodes[:, :-1])
    )

    # Remove dummy node features
    output_pnodes = output_pnodes[:, :-1, :]

    return output_pnodes

  def call(self,
    inputs: Inputs,
    graphs: RegionInteractionGraphSet,
    key: flax.typing.PRNGKey = None,
  ) -> Array:
    """Inputs must be of shape [batch_size, 1, num_physical_nodes, num_inputs]"""

    # Check input functions
    self._check_function(inputs.u, x=inputs.x_inp)
    if inputs.c is not None:
      self._check_function(inputs.c, x=inputs.x_inp)
    assert inputs.u.shape[3] == self.num_outputs

    # Read dimensions
    batch_size = inputs.u.shape[0]
    num_pnodes_inp = inputs.x_inp.shape[2]
    num_pnodes_out = inputs.x_out.shape[2]

    # Prepare the time channel
    if self.concatenate_t:
      assert inputs.t is not None
      t_inp = jnp.array(inputs.t, dtype=jnp.float32)
      if t_inp.ndim == 4:
        t_inp = t_inp[:, :, 0, 0]
      if t_inp.size == 1:
        t_inp = jnp.tile(t_inp.reshape(1, 1), reps=(batch_size, 1))
    # Prepare the time difference channel
    if self.concatenate_tau:
      assert inputs.tau is not None
      tau = jnp.array(inputs.tau, dtype=jnp.float32)
      if tau.ndim == 4:
        tau = tau[:, :, 0, 0]
      if tau.size == 1:
        tau = jnp.tile(tau.reshape(1, 1), reps=(batch_size, 1))
    else:
      tau = None

    # Concatenate the known coefficients to the channels of the input function
    if inputs.c is None:
      u_inp = inputs.u
    else:
      u_inp = jnp.concatenate([inputs.u, inputs.c], axis=-1)

    # Prepare the physical node features
    # u -> [batch_size, num_pnodes_inp, num_inputs]
    pnode_features = jnp.moveaxis(u_inp,
      source=(0, 1, 2, 3), destination=(0, 3, 1, 2)
    ).squeeze(axis=3)

    # Concatente with forced features
    pnode_features_forced = []
    if self.concatenate_t:
      pnode_features_forced.append(jnp.tile(jnp.expand_dims(t_inp, axis=1), reps=(1, num_pnodes_inp, 1)))
    if self.concatenate_tau:
      pnode_features_forced.append(jnp.tile(jnp.expand_dims(tau, axis=1), reps=(1, num_pnodes_inp, 1)))
    pnode_features = jnp.concatenate([pnode_features, *pnode_features_forced], axis=-1)

    # Run the GNNs
    subkey, key = jax.random.split(key) if (key is not None) else (None, None)
    output_pnodes = self._encode_process_decode(
      graphs=graphs, pnode_features=pnode_features, tau=tau, key=subkey)

    # Reshape the output to u
    # [batch_size, num_pnodes_out, num_outputs] -> [batch_size, 1, num_pnodes_out, num_outputs]
    output = self._prepare_features(output_pnodes)
    self._check_function(output, x=inputs.x_out)

    return output

def _subsample_pointset(key, x: Array, factor: float) -> Array:
  """Downsamples a point cloud by randomly subsampling them"""
  x = jnp.array(x)
  x_shuffled, = shuffle_arrays(key, [x])
  return x_shuffled[:int(x.shape[0] / factor)]

def _upsample_pointset(key, x: Array, factor: float) -> Array:
  """Upsamples a point cloud by adding the middle point of randomly selected simplices."""

  factor = factor ** x.shape[-1]
  num_new_points = int(x.shape[0] * (factor - 1))
  tri = Delaunay(points=x)
  simplices = jax.random.permutation(key=key, x=tri.simplices)[jnp.arange(num_new_points)]
  x_ext = np.mean(x[simplices], axis=1)
  return np.concatenate([x, x_ext], axis=0)

def _get_edges_from_triangulation(tri: Delaunay, bidirectional: bool = True):
  indptr, cols = tri.vertex_neighbor_vertices
  rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
  edges = np.stack([rows, cols], -1)
  if bidirectional:
    edges = np.concatenate([edges, np.flip(edges, axis=-1)], axis=0)
  return edges

def _compute_triangulation_medians(tri: Delaunay) -> Array:
  # Only in 2D

  edges = np.zeros(shape=tri.simplices.shape)
  medians = np.zeros(shape=tri.simplices.shape)
  for i in range(tri.simplices.shape[1]):
    points = tri.points[np.delete(tri.simplices, i, axis=1)]
    points = [p.squeeze(1) for p in np.split(points, axis=1, indices_or_sections=2)]
    edges[:, i] = np.linalg.norm(np.subtract(*points), axis=1)
  for i in range(tri.simplices.shape[1]):
    medians[:, i] = .67 * np.sqrt((2 * np.sum(np.power(np.delete(edges, i, axis=1), 2), axis=1) - np.power(edges[:, i], 2)) / 4)

  return medians
