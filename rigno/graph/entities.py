# Adopted from https://github.com/google-deepmind/graphcast
# Accessed on 16 February 2024, commit 8debd7289bb2c498485f79dbd98d8b4933bfc6a7
# Codes are slightly modified to be compatible with Flax
#
# @article{lam2023learning,
#   title={Learning skillful medium-range global weather forecasting},
#   author={Lam, Remi and Sanchez-Gonzalez, Alvaro and Willson, Matthew and Wirnsberger, Peter and Fortunato, Meire and Alet, Ferran and Ravuri, Suman and Ewalds, Timo and Eaton-Rosen, Zach and Hu, Weihua and others},
#   journal={Science},
#   volume={382},
#   number={6677},
#   pages={1416--1421},
#   year={2023},
#   publisher={American Association for the Advancement of Science}
# }

"""Data-structure for storing graphs with typed edges and nodes."""

from typing import NamedTuple, Any, Union, Tuple, Mapping

ArrayLike = Union[Any]  # np.ndarray, jnp.ndarray, tf.tensor
ArrayLikeTree = Union[Any, ArrayLike]  # Nest of ArrayLike

# All tensors have a leading `batch_axis` of shape `bsz`

class NodeSet(NamedTuple):
  """Represents a set of nodes."""
  n_node: ArrayLike  # [bsz, 1]
  features: ArrayLikeTree  # [bsz, n_node, n_feats]

class EdgesIndices(NamedTuple):
  """Represents indices to nodes adjacent to the edges."""
  senders: ArrayLike  # [bsz, n_edge]
  receivers: ArrayLike  # [bsz, n_edge]

class EdgeSet(NamedTuple):
  """Represents a set of edges."""
  n_edge: ArrayLike  # [bsz, 1]
  indices: EdgesIndices
  features: ArrayLikeTree  # [bsz, n_edge, n_feats]

class Context(NamedTuple):
  # `n_graph` always contains ones but it is useful to query the leading shape
  # in case of graphs without any nodes or edges sets.
  n_graph: ArrayLike  # [bsz, 1]
  features: ArrayLikeTree  # [bsz, n_feats]

class EdgeSetKey(NamedTuple):
  # Name of the EdgeSet
  name: str
  # Sender node set name and receiver node set name connected by the edge set
  node_sets: Tuple[str, str]

class TypedGraph(NamedTuple):
  """A graph with typed nodes and edges.

  A typed graph is made of a context, multiple sets of nodes and multiple
  sets of edges connecting those nodes (as indicated by the EdgeSetKey).
  """

  context: Context
  nodes: Mapping[str, NodeSet]
  edges: Mapping[EdgeSetKey, EdgeSet]

  def edge_key_by_name(self, name: str) -> EdgeSetKey:
    found_key = [k for k in self.edges.keys() if k.name == name]
    if len(found_key) != 1:
      raise KeyError('invalid edge key "{}". Available edges: [{}]'.format(
        name, ', '.join(x.name for x in self.edges.keys())))
    return found_key[0]

  def edge_by_name(self, name: str) -> EdgeSet:
    return self.edges[self.edge_key_by_name(name)]
