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

"""JAX implementation of Graph Networks Simulator."""

from typing import Mapping, Optional, Union, Callable

import jax
import jax.numpy as jnp
import flax.linen as nn
import jraph

from rigno.graph.entities import TypedGraph
from rigno.graph.networks import GraphMapFeatures, InteractionNetwork
from rigno.models.utils import FeedForwardBlock


class DeepTypedGraphNet(nn.Module):
  """Deep Graph Neural Network.

  It works with TypedGraphs with typed nodes and edges. It runs message
  passing on all of the node sets and all of the edge sets in the graph. For
  each message passing step a `typed_graph_net.InteractionNetwork` is used to
  update the full TypedGraph by using different MLPs for each of the node sets
  and each of the edge sets.

  If embed_{nodes,edges} is specified the node/edge features will be embedded
  into a fixed dimensionality before running the first step of message passing.

  If {node,edge}_output_size the final node/edge features will be embedded into
  the specified output size.

  This class may be used for shared or unshared message passing:
  * num_message_passing_steps = N, num_processor_repetitions = 1, gives
    N layers of message passing with fully unshared weights:
    [W_1, W_2, ... , W_M] (default)
  * num_message_passing_steps = 1, num_processor_repetitions = M, gives
    M layers of message passing with fully shared weights:
    [W_1] * M
  * num_message_passing_steps = N, num_processor_repetitions = M, gives
    M*N layers of message passing with both shared and unshared message passing
    such that the weights used at each iteration are:
    [W_1, W_2, ... , W_N] * M

  Args:
    node_latent_size: Size of the node latent representations.
    edge_latent_size: Size of the edge latent representations.
    mlp_num_hidden_layers: Number of hidden layers in all MLPs.
    num_message_passing_steps: Number of unshared message passing steps
        in the processor steps.
    num_processor_repetitions: Number of times that the same processor is
        applied sequencially.
    embed_nodes: If False, the node embedder will be omitted.
    embed_edges: If False, the edge embedder will be omitted.
    node_output_size: Size of the output node representations for
        each node type. For node types not specified here, the latent node
        representation from the output of the processor will be returned.
    edge_output_size: Size of the output edge representations for
        each edge type. For edge types not specified here, the latent edge
        representation from the output of the processor will be returned.
    include_sent_messages_in_node_update: Whether to include pooled sent
        messages from each node in the node update.
    use_layer_norm: Whether to use LayerNorm layers.
    conditioned_normalization: Whether to use conditioned normalization layers.
    cond_norm_hidden_size: Hidden size for the shallow MLP used for
      computing shift and scales in the conditioned normalization layers.
    activation: name of the activation function.
    f32_aggregation: Use float32 in the edge aggregation.
    aggregate_edges_for_nodes_fn: function used to aggregate messages
      to each node.
    name: Name of the model.
  """

  node_latent_size: Mapping[str, int]
  edge_latent_size: Mapping[str, int]
  mlp_num_hidden_layers: int
  num_message_passing_steps: int
  num_processor_repetitions: int = 1
  embed_nodes: bool = True
  embed_edges: bool = True
  node_output_size: Optional[Mapping[str, int]] = None
  edge_output_size: Optional[Mapping[str, int]] = None
  include_sent_messages_in_node_update: bool = False
  use_layer_norm: bool = True
  conditioned_normalization: bool = False
  cond_norm_hidden_size: int = 16
  activation: str = 'swish'
  f32_aggregation: bool = False
  aggregate_edges_for_nodes_fn: Callable = jraph.segment_mean

  def setup(self):
    self._activation = _get_activation_fn(self.activation)

    # The embedder graph network independently embeds edge and node features.
    if self.embed_edges:
      embed_edge_fn = {
        edge_set_name: FeedForwardBlock(
          layer_sizes=(
            [self.edge_latent_size[edge_set_name]] * self.mlp_num_hidden_layers
            + [self.edge_latent_size[edge_set_name]]
          ),
          activation=self._activation,
          use_layer_norm=self.use_layer_norm,
          use_conditional_norm=False,
          name=f'encoder_edges_{edge_set_name}',
        )
        for edge_set_name in self.edge_latent_size.keys()
      }
    else:
      embed_edge_fn = None
    if self.embed_nodes:
      embed_node_fn = {
        node_set_name: FeedForwardBlock(
          layer_sizes=(
            [self.node_latent_size[node_set_name]] * self.mlp_num_hidden_layers
            + [self.node_latent_size[node_set_name]]
          ),
          activation=self._activation,
          use_layer_norm=self.use_layer_norm,
          use_conditional_norm=False,
          name=f'encoder_nodes_{node_set_name}',
        )
        for node_set_name in self.node_latent_size.keys()
      }
    else:
      embed_node_fn = None
    embedder_kwargs = dict(
        embed_edge_fn=embed_edge_fn,
        embed_node_fn=embed_node_fn,
    )
    self._embedder_network = GraphMapFeatures(**embedder_kwargs)

    if self.f32_aggregation:
      def aggregate_fn(data, *args, **kwargs):
        dtype = data.dtype
        data = data.astype(jnp.float32)
        output = self.aggregate_edges_for_nodes_fn(data, *args, **kwargs)
        output = output.astype(dtype)
        return output

    else:
      def aggregate_fn(data, *args, **kwargs):
        output = self.aggregate_edges_for_nodes_fn(data, *args, **kwargs)
        return output

    # Create `num_message_passing_steps` graph networks with unshared parameters
    # that update the node and edge latent features.
    # Note that we can use `InteractionNetwork` because
    # it also outputs the messages as updated edge latent features.
    self._processor_networks = [
      InteractionNetwork(
        update_edge_fn={
          edge_set_name: FeedForwardBlock(
            layer_sizes=(
              [self.edge_latent_size[edge_set_name]] * self.mlp_num_hidden_layers
              + [self.edge_latent_size[edge_set_name]]
            ),
            activation=self._activation,
            use_layer_norm=self.use_layer_norm,
            use_conditional_norm=self.conditioned_normalization,
            cond_norm_hidden_size=self.cond_norm_hidden_size,
            name=f'processor_{step_i}_edges_{edge_set_name}',
          )
          for edge_set_name in self.edge_latent_size.keys()
        },
        update_node_fn={
          node_set_name: FeedForwardBlock(
            layer_sizes=(
              [self.node_latent_size[node_set_name]] * self.mlp_num_hidden_layers
              + [self.node_latent_size[node_set_name]]
            ),
            activation=self._activation,
            use_layer_norm=self.use_layer_norm,
            use_conditional_norm=self.conditioned_normalization,
            cond_norm_hidden_size=self.cond_norm_hidden_size,
            name=f'processor_{step_i}_nodes_{node_set_name}',
          )
          for node_set_name in self.node_latent_size.keys()
        },
        aggregate_edges_for_nodes_fn=aggregate_fn,
        include_sent_messages_in_node_update=self.include_sent_messages_in_node_update,
      )
      for step_i in range(self.num_message_passing_steps)
    ]

    # The output MLPs converts edge/node latent features into the output sizes.
    output_kwargs = dict(
      embed_edge_fn={
        edge_set_name: FeedForwardBlock(
          layer_sizes=(
            [self.edge_output_size[edge_set_name]] * self.mlp_num_hidden_layers
            + [self.edge_output_size[edge_set_name]]
          ),
          activation=self._activation,
          use_layer_norm=False,
          use_conditional_norm=False,
          name=f'decoder_edges_{edge_set_name}',
        )
        for edge_set_name in self.edge_output_size.keys()
      } if self.edge_output_size else None,
      embed_node_fn={
        node_set_name: FeedForwardBlock(
          layer_sizes=(
            [self.node_output_size[node_set_name]] * self.mlp_num_hidden_layers
            + [self.node_output_size[node_set_name]]
          ),
          activation=self._activation,
          use_layer_norm=False,
          use_conditional_norm=False,
          name=f'decoder_nodes_{node_set_name}',
        )
        for node_set_name in self.node_output_size.keys()
      } if self.node_output_size else None,
    )
    self._output_network = GraphMapFeatures(**output_kwargs)

  def __call__(self, input_graph: TypedGraph, condition: Union[None, float]) -> TypedGraph:
    """Forward pass of the learnable dynamics model."""

    # Embed input features (if applicable).
    latent_graph_0 = self._embed(input_graph, c=condition)

    # Do `m` message passing steps in the latent graphs.
    latent_graph_m = self._process(latent_graph_0, c=condition)

    # Compute outputs from the last latent graph (if applicable).
    return self._output(latent_graph_m, c=None)

  def _embed(self, input_graph: TypedGraph, **kwargs) -> TypedGraph:
    """Embeds the input graph features into a latent graph."""

    # Copy the context to all of the node types, if applicable.
    context_features = input_graph.context.features
    if jax.tree_util.tree_leaves(context_features):
      # This code assumes a single input feature array for the context and for
      # each node type.
      assert len(jax.tree_util.tree_leaves(context_features)) == 1
      new_nodes = {}
      for node_set_name, node_set in input_graph.nodes.items():
        node_features = node_set.features
        broadcasted_context = jnp.repeat(
            context_features, node_set.n_node, axis=0,
            total_repeat_length=node_features.shape[0])
        new_nodes[node_set_name] = node_set._replace(
            features=jnp.concatenate(
                [node_features, broadcasted_context], axis=-1))
      input_graph = input_graph._replace(
          nodes=new_nodes,
          context=input_graph.context._replace(features=()))

    # Embeds the node and edge features.
    latent_graph_0 = self._embedder_network(input_graph, **kwargs)
    return latent_graph_0

  def _process(self, latent_graph_0: TypedGraph, **kwargs) -> TypedGraph:
    """Processes the latent graph with several steps of message passing."""

    # Do `num_message_passing_steps` with each of the `self._processor_networks`
    # with unshared weights, and repeat that `self._num_processor_repetitions`
    # times.
    latent_graph = latent_graph_0
    # TODO: OPTIMIZATION: use jax.lax.fori_loop
    for _ in range(self.num_processor_repetitions):
      # NOTE: OPTIMIZATION: using jax.lax.fori_loop is not possible for this loop
      for processor_network in self._processor_networks:
        latent_graph = self._process_step(processor_network, latent_graph, **kwargs)

    return latent_graph

  def _process_step(self, processor_network_k, latent_graph_prev_k: TypedGraph, **kwargs) -> TypedGraph:
    """Single step of message passing with node/edge residual connections."""

    # One step of message passing.
    latent_graph_k = processor_network_k(latent_graph_prev_k, **kwargs)

    # Add residuals.
    nodes_with_residuals = {}
    for k, prev_set in latent_graph_prev_k.nodes.items():
      nodes_with_residuals[k] = prev_set._replace(
          features=prev_set.features + latent_graph_k.nodes[k].features)

    edges_with_residuals = {}
    for k, prev_set in latent_graph_prev_k.edges.items():
      edges_with_residuals[k] = prev_set._replace(
          features=prev_set.features + latent_graph_k.edges[k].features)

    latent_graph_k = latent_graph_k._replace(
        nodes=nodes_with_residuals, edges=edges_with_residuals)
    return latent_graph_k

  def _output(self, latent_graph: TypedGraph, **kwargs) -> TypedGraph:
    """Produces the output from the latent graph."""

    return self._output_network(latent_graph, **kwargs)

def _get_activation_fn(name):
  """Return activation function corresponding to function_name."""

  if name == 'identity':
    return lambda x: x
  if hasattr(jax.nn, name):
    return getattr(jax.nn, name)
  if hasattr(jnp, name):
    return getattr(jnp, name)

  raise ValueError(f'Unknown activation function {name} specified.')
