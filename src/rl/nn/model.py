from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import flax.nnx as nnx

@jax.tree_util.register_dataclass
@dataclass
class Args:
    dims: int = 32
    inner_dims: int = 128

    n_input: int = 29
    n_output: int = 12

    dtype = jax.dtypes.bfloat16

def linear(x: jax.Array, w: jax.Array, bias: Optional[jax.Array] = None):
    """

    """
    x = jnp.matmul(x, w)
    if bias != None:
        x = x + bias
    return x

class RMSNorm(nnx.Module):
    """
        Root Mean Square Normalization

        Attributes:
            dims (int): Dimension of input
            eps (float): Epsilon value for numerical stability
    """
    def __init__(self, dims: int, param_dtype=jnp.float32, eps: float = 1e-8):
        """
            Initializer for Root Mean Square Module

            Args:
                dims (int): Dimension of input
                eps (float): Epsilon value for numerical stability
        """
        self.dims = dims
        self.eps = eps

        self.w = nnx.Param(
            jnp.zeros((dims,), dtype=param_dtype)
        )

    def __call__(self, x: jax.Array):
        denom = jax.lax.rsqrt(
            jnp.mean(x**2, axis=-1) + self.eps
        )
        x = x * denom
        x = x * (1 + self.w.value)
        return x

class Linear(nnx.Module):
    """
        Linear Layer. Performs matrix multiplication with trainable weights
    """
    def __init__(
        self, 
        in_dims: int, 
        out_dims: int, 
        *,
        use_bias: bool = False,
        kernel_initializer = nnx.initializers.he_normal(),
        bias_initializer = nnx.initializers.zeros_init(),
        param_dtype = jnp.float32,
        key: jax.random.PRNGKey,
        **meta
    ):
        """
            Initializer for Linear layer

            Args:
                in_dims (int): Input dimensions
                out_dims (int): Output dimensions
                use_bias (bool): Specifies whether to use bias or not
                
        """
        self.w = nnx.Param(
            kernel_initializer(
                key=key, 
                shape=(in_dims, out_dims), 
                dtype=param_dtype, 
                out_sharding=meta["sharding"]
            )
        )

        self.bias = None
        if use_bias:
            self.bias = nnx.Param(
                bias_initializer(key, (out_dims,), dtype=param_dtype)
            )

    def __call__(self, x: jax.Array):
        return linear(x, self.w.value, self.bias.value)


class MLP(nnx.Module):
    """
        Multi-Layer Perceptron Module
    """
    def __init__(
        self, 
        in_dims: int,
        inner_dims: int,
        out_dims: int, 
        activation = jax.nn.gelu,
        *,
        param_dtype = jnp.float32,
        key: jax.random.PRNGKey,
    ):
        _, k_l1, k_l2, k_l3 = jax.random.split(key, 4)
        
        self.activation = activation

        # Linear functions
        self.l1 = Linear(in_dims, inner_dims, param_dtype=param_dtype, key=k_l1)
        self.l2 = Linear(in_dims, inner_dims, param_dtype=param_dtype, key=k_l2)
        self.l3 = Linear(in_dims, out_dims, param_dtype=param_dtype, key=k_l3)

        # Activation returns same value when initialized as None
        if activation == None:
            activation = lambda x: x

    def __call__(self, x: jax.Array):
        x1 = self.l1(x)
        x2 = self.l2(x)
        # Apply activation
        x1 = self.activation(x1)
        x = x1 * x2
        x = self.l3(x)
        return x

class ControllerNet(nnx.Module):
    """
        Controller Network Module
    """
    def __init__(self, args: Args, key: jax.random.PRNGKey):
        self.args = args

        _, k_proj, k_mlp, k_head = jax.random.split(key, 4)

        self.proj = Linear(
            args.n_input,
            args.n_output,
            key=k_proj
        )

        self.norm = RMSNorm(args.dims, args.dtype)

        self.mlp = MLP(
            args.n_input,
            args.inner_dims,
            args.n_output,
            activation=jax.nn.gelu,
            key=k_mlp
        )

        self.head = Linear(
            args.n_input,
            args.n_output,
            use_bias=True,
            key=k_head
        )

    def __call__(self, x: jax.Array):
        x = self.proj(x)
        x = self.mlp(self.norm(x))
        x = self.head(x)
        return x

class ActorCritic(nnx.Module):
    """
    Couples Actor and Critic models for optimization
    """
    def __init__(
        self,
        actor: nnx.Module,
        critic: nnx.Module
    ):
        self.actor = actor
        self.critic = critic

    def __call__(self, state: jax.Array):
        pdf_params = self.actor(state)
        value = self.critic(state)
        return pdf_params, value
    