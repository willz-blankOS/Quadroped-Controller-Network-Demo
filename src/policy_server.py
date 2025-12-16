"""
    Client TCP server for Isaac sim bridge
"""
import json

import socket
import zmq
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.nnx as nnx

import optax

from rl.nn.model import ActorCritic, ControllerNet, MLP, Args
from rl.policy.policy import Policy, Curriculum
from rl.saving.saving import TrainState, PolicyCheckpointManager
from rl.data.data import DataBuffer
from rl.misc import stats

from rich.progress import track

HOST = "127.0.0.1"
PORT = 50007

EPISODES = 1000
STEP_DIMS = 12

def flatten_state(state: dict):
    return jnp.concatenate([
        jnp.array(state["joint_pos"], dtype=jnp.float32),
        jnp.array(state["joint_vel"], dtype=jnp.float32),
        jnp.array(state["base_rpy"], dtype=jnp.float32),
        jnp.array(state["base_lin_vel"], dtype=jnp.float32),
        jnp.array(state["base_ang_vel"], dtype=jnp.float32)
    ])

def reset(sock: zmq.Socket):
    sock.send_json({"type": "reset"})
    msg = sock.recv_json()
    state = msg["state"]
    state_vec = flatten_state(state)
    
    return state_vec

def convert_to_jax(state: dict[bool, bytes]):
    processed_jax: dict[jax.Array] = {}
    for key, val in state:
        arr = jnp.frombuffer(val)
        idx = jnp.arange(0, arr.shape[0])
        processed_jax[key] = arr[idx]

    return processed_jax

def step(sock: zmq.Socket, actions: jax.Array):
    sock.send_json({
        "type": "step",
        "actions": actions.tolist()
    })
    msg = sock.recv_json()
    state = msg["state"]
    state = flatten_state(state)
    return state

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", "-c", type=str, required=True)
    parser.add_argument("--episodes", "-e", type=int, required=False, default=1_000)
    parser.add_argument("--steps", "-s", type=int, required=False, default=128)
    curr_group = parser.add_mutually_exclusive_group(required=False)
    curr_group.add_argument("--balance", action="store_true"),
    curr_group.add_argument("--walk", action="store_true"),
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    EPISODES = int(args.episodes)
    STEPS = int(args.steps)

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{HOST}:{PORT}")

    socket.send(json.dumps({"type": "connect"}).encode())

    # Get Checkpoints
    ckpt_manager = PolicyCheckpointManager(
        checkpoint_dir
    )

    train_state, meta = ckpt_manager.maybe_restore()

    # Create model
    rng = nnx.Rngs()
    args = Args()
    actor = ControllerNet(args, rng())
    critic = MLP(args.n_input, args.dims, 1)
    model = ActorCritic(actor, critic)

    # Optimizer
    optimizer = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_learning_rate(3e-4),
        optax.scale(-1)
    )
    optimizer = nnx.Optimizer(model, optimizer, wrt=nnx.Param)

    if train_state == None:
        train_state = TrainState(
            nnx.state(optimizer),
            nnx.state(model),
            0
        )
    else:
        model_graphdef, _ = nnx.split(model)
        model = nnx.merge(model_graphdef, train_state.model_state)

        optimizer_graphdef, _ = nnx.split(optimizer)
        optimizer = nnx.merge(optimizer_graphdef, train_state.optimizer_state)

    buffer = DataBuffer(STEPS)

    state = reset(socket)
    end_condition = False
    for episode in track(
        range(EPISODES), 
        description="Training quadroped...", 
        total=EPISODES
    ):
        flattened_state = flatten_state(state)

        step_count = 0
        while not end_condition:
            value = model.critic(state)

            if obs != dict():
                obs["future_value"] = value

                # Buffer update before 
                buffer.update(obs)
                obs.clear()

                if step_count == 0:
                    # Update
                    ...
            
            actions = model.actor(state)
            sampled_actions = stats.sample_gauss(actions[:,0], actions[:,1])
            pi = stats.log_pdf_guass(sampled_actions, actions[:,0], actions[:,1])
            obs = {
                "state": state, 
                "value": value,
                "actions": sampled_actions, # Use sampled action
                "pi": pi, # Action likelihood
                "rewards": ... # We need to handle rewards solver first
            }

            state = step(socket, actions)

            train_state.steps += 1
            train_state.model_state = nnx.state(model)
            train_state.optimizer_state = nnx.state(optimizer)
            ckpt_manager.maybe_save_periodic(train_state, {})

            step_count += 1