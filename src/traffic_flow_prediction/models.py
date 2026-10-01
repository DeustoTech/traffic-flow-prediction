from __future__ import annotations

from enum import Enum

import jax
import jax.numpy as jnp
import jraph
from flax import nnx
from jax.typing import ArrayLike


class GATLayer(nnx.Module):

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        *,
        negative_slope: float = 0.2,
        rngs:           nnx.Rngs,
    ) -> None:
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs)
        self.att_src = nnx.Linear(out_features, 1, use_bias=False, rngs=rngs)
        self.att_dst = nnx.Linear(out_features, 1, use_bias=False, rngs=rngs)

        self.negative_slope = negative_slope

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:

        h = self.linear(graph.nodes)                  # (N,F)

        senders = graph.senders
        receivers = graph.receivers

        h_src = h[senders]                            # (E,F)
        h_dst = h[receivers]                          # (E,F)

        # e_ij = a_l^T Wh_i + a_r^T Wh_j
        e = self.att_src(h_src) + self.att_dst(h_dst) # (E,1)
        e = nnx.leaky_relu(e, self.negative_slope)
        e = e.squeeze(-1)

        alpha = jraph.segment_softmax(e, receivers, num_segments=h.shape[0])
        messages = alpha[:, None] * h_src
        out = jraph.segment_sum(messages, receivers, num_segments=h.shape[0])

        return graph._replace(nodes=out)


class GCNLayer(nnx.Module):

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        *,
        use_bias: bool = True,
        rngs:     nnx.Rngs,
    ) -> None:
        self.linear = nnx.Linear(in_features, out_features, use_bias=use_bias, rngs=rngs)

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        nodes = graph.nodes
        sum_n_node = nodes.shape[0]

        nodes = self.linear(nodes)

        senders   = jnp.concatenate([graph.senders,   jnp.arange(sum_n_node)])
        receivers = jnp.concatenate([graph.receivers, jnp.arange(sum_n_node)])
        
        def count_edges(x: ArrayLike) -> jax.Array:
            return jraph.segment_sum(jnp.ones_like(x), x, num_segments=sum_n_node)
        
        sender_degree   = count_edges(senders)
        receiver_degree = count_edges(receivers)
        
        nodes = nodes * jax.lax.rsqrt(jnp.maximum(sender_degree, 1.0))[:, jnp.newaxis]
        nodes = jraph.segment_sum(nodes[senders], receivers, num_segments=sum_n_node)
        nodes = nodes * jax.lax.rsqrt(jnp.maximum(receiver_degree, 1.0))[:, jnp.newaxis]

        return graph._replace(nodes=nodes)


class ReLULayer(nnx.Module):
    
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        return graph._replace(nodes=nnx.relu(graph.nodes))


class SpatialModule(nnx.Module):
    
    def __init__(
        self,
        in_features:     int,
        hidden_features: int,
        out_features:    int,
        num_layers:      int,
        *,
        layer: SpatialType,
        rngs:  nnx.Rngs,
    ) -> None:
        self.layers: list[nnx.Module] = []
        for i in range(num_layers):
            in_feat  = in_features  if i == 0              else hidden_features
            out_feat = out_features if i == num_layers - 1 else hidden_features
            self.layers.append(layer.value(in_feat, out_feat, rngs=rngs))
            self.layers.append(ReLULayer())
    
    def __call__(self, graph: jraph.GraphsTuple) -> jax.Array:

        def run_layers(nodes):
            g = graph._replace(nodes=nodes)
            for layer in self.layers:
                g = layer(g)
            return g.nodes

        x = graph.nodes             # (N,T,F)
        x = jnp.swapaxes(x, 0, 1)   # (T,N,F)
        x = nnx.vmap(run_layers)(x) # (T,N,H)
        x = jnp.swapaxes(x, 0, 1)   # (N,T,H)
        return x


class TemporalModule(nnx.Module):
    
    def __init__(
        self,
        in_features:     int,
        hidden_features: int,
        in_time:         int,
        out_time:        int,
        *,
        autoregressive: bool = False,
        cell: TemporalType,
        rngs: nnx.Rngs,
    ) -> None:
        self.out_time = out_time
        self.encoder = nnx.Bidirectional(
            nnx.RNN(cell.value(in_features, hidden_features, rngs=rngs)),
            nnx.RNN(cell.value(in_features, hidden_features, rngs=rngs)),
            merge_fn=jnp.add,
        )
        if autoregressive:
            self.decoder = cell.value(hidden_features, hidden_features, rngs=rngs)
        else:
            self.decoder = nnx.Linear(in_time, out_time, rngs=rngs)
        self.autoregressive = autoregressive
        self.out = nnx.Linear(hidden_features, 1, rngs=rngs)
    
    def encode(self, x: ArrayLike) -> jax.Array:
        # x                 # (N,T,H)
        h = self.encoder(x) # (N,T,H)
        return h
    
    def decode(self, h: ArrayLike) -> jax.Array:
        if self.autoregressive:
            carry_0 = self.decoder.initialize_carry(h.shape)
                    
            def run_decoder(state, _):
                carry, x = state
                carry, y = self.decoder(carry, x)
                return (carry, y), y
            
            scan_fn = nnx.scan(
                run_decoder,
                length=self.out_time,
                in_axes=(nnx.Carry, 0),
                out_axes=(nnx.Carry, 0),
            )
            
            # h                                # (N,T,H)
            h = h[:, -1, :]                    # (N,H)
            _, y = scan_fn((carry_0, h), None) # (P,N,H)
            y = jnp.swapaxes(y, 0, 1)          # (N,P,H)
        else:
            # h                       # (N,T,H)
            h = jnp.swapaxes(h, 1, 2) # (N,H,T)
            y = self.decoder(h)       # (N,H,P)
            y = jnp.swapaxes(y, 1, 2) # (N,P,H)
        
        y = self.out(y) # (N,P,1)
        return y
        
    
    def __call__(self, x: ArrayLike) -> jax.Array:
        h = self.encode(x)
        y = self.decode(h)
        return y


class GNN(nnx.Module):

    def __init__(
        self,
        in_features:     int,
        hidden_features: int,
        num_layers:      int,
        in_time:         int,
        out_time:        int,
        *,
        spatial:  SpatialType,
        temporal: TemporalType,
        rngs:     nnx.Rngs,
    ):
        self.spatial = SpatialModule(
            in_features,
            hidden_features,
            hidden_features,
            num_layers,
            layer=spatial,
            rngs=rngs,
        )
        self.temporal = TemporalModule(
            hidden_features,
            hidden_features,
            in_time,
            out_time,
            cell=temporal,
            rngs=rngs,
        )

    def __call__(self, graph: jraph.GraphsTuple) -> jax.Array:
        x = self.spatial(graph)
        y = self.temporal(x)
        return y


class SpatialType(Enum):
    GAT = GATLayer
    GCN = GCNLayer


class TemporalType(Enum):
    LSTM = nnx.LSTMCell
    GRU  = nnx.GRUCell