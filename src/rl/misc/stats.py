import jax
import jax.numpy as jnp

# ===========================
# Distributions
# ===========================

# ---------------------
# 2-Parameter Gaussian
# ---------------------

def sample_gauss(loc: jax.Array, scale: jax.Array):
    ...

def pdf_gauss(x: jax.Array, loc: jax.Array, scale: jax.Array):
    s = jax.lax.rsqrt(
        2 * jnp.pi * scale
    )
    e = jnp.exp(
        -(x - loc)**2 / (2 * scale)**2
    )

    likelihood = s * e
    return likelihood

def log_pdf_guass(x: jax.Array, loc: jax.Array, scale: jax.Array):
    return jnp.log(pdf_gauss(x, loc, scale))