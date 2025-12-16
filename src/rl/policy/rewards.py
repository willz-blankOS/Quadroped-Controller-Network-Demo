import jax
import jax.numpy as jnp

class RewardGenerator:
    def __init__(self):
        pass

    def __call__(self, *args, **kwds):
        pass


def euclidean_distance(x: jax.Array, y: jax.Array):
    diff_sq = (x - y)**2
    distance = jax.lax.sqrt(diff_sq.sum(axis=-1))
    return distance


def simple_find_reward_function(
        pose: jax.Array, 
        prev_pose: jax.Array,
        target_pose: jax.Array,
        step_cost: jax.Array | float = 0.01,
        success_threshold: jax.Array | float = 0.1,
        sucess_bonus: float = 5.0,
        **kwargs
    ):
    # Slice theta out of pose
    pose = pose[:-1]
    prev_pose = prev_pose[:-1]
    
    current_dist = euclidean_distance(pose, target_pose)
    prev_dist = euclidean_distance(prev_pose, target_pose)

    progress = current_dist - prev_dist

    # Success on near target and essentially stopped
    sucess = current_dist < success_threshold and progress < 0.01 
    bonus = jnp.where(sucess, sucess_bonus, 0)

    return progress + bonus - step_cost