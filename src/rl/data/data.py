import jax
import jax.numpy as jnp

class DataBuffer:
    def __init__(self, size: jax.Array):
        self.size = size
        self.prev_buffer: dict[jax.Array] = {}
        self.buffer: dict[jax.Array] = {}

    def update(self, update: dict[jax.Array]):
        """
            TODO: We need to make this handle cases where an undate to a key in buffer isnt provided
        """
        for key, item in update:
            if key not in self.buffer:
                self.buffer[key] = jnp.repeat(item, self.size)
                self.prev_buffer[key] = jnp.repeat(item, self.size)
            else:
                buff_arr = self.buffer[key]
                buff_arr = jnp.roll(buff_arr, -1, axis=0)
                old = buff_arr[-1]
                buff_arr = buff_arr.at[0].set(item)

                prev_arr = self.prev_buffer[key]
                prev_arr = jnp.roll(prev_arr, -1, axis=0)
                prev_arr = prev_arr.at[-1].set(old)
