from typing import Callable, List, Optional

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np

def estimate_gae_advatage(obs: dict[str, jax.Array], discount: float, lam: float):
    """
        Uses GAE to estimate advantage
    """

    rewards = obs["rewards"]
    advantage = jnp.zeros_like(rewards)
    value = obs["value"]
    mask = 1.0 - obs["done"]

    gae = 0.0
    T = rewards.shape[0]
    for t in range(T - 1, -1, -1):
        mask_t = mask[t]
        delta = rewards[t] + discount * value[t+1] * mask_t - value[t]
        gae = delta + discount * lam * mask_t * gae
        advantage = advantage.at[t].set(gae)

    value_targets = advantage + value[:-1]
    return advantage, value_targets

def compute_clipped_ppo(
        new_pi: jax.Array,
        old_pi: jax.Array,
        advantage: jax.Array,
        *,
        eps: float = 0.2,
        ratio_eps: float = 1e-8
):
    ratio = new_pi / (old_pi + ratio_eps)
    loss = jnp.minimum(
        ratio * advantage, jnp.clip(ratio, 1 - eps, 1 + eps) * advantage
    ).mean()
    return loss

def gaussian_entropy(pi: jax.Array):
    entropy = jnp.sum(
        pi * jnp.log(jnp.maximum(pi, 1e-8)), 
        axis=-1
    )
    return -entropy
