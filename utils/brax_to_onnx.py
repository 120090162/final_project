"""Convert Brax PPO networks to ONNX format."""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORMS"] = "cuda,cpu"

import functools
from pathlib import Path

from absl import app
from absl import flags
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.checkpoint import load
from etils import epath
import jax
import jax.numpy as jp
from mujoco_playground import registry
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params
import mujoco_playground
import numpy as np
import onnxruntime as rt
import tensorflow as tf
from tensorflow.keras import layers  # type: ignore
import tf2onnx

from params import _ONNX_DIR

if __name__ == "__main__":
    import sys

    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
    # Add the current directory to Python path to find module
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import envs
import envs.params as env_params

_CHECKPOINT_PATH = flags.DEFINE_string(
    "checkpoint_path",
    None,
    "Path to the checkpoint directory",
)
_ENV_NAME = flags.DEFINE_string(
    "env_name",
    None,
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    None,
    "Output path for the ONNX model. Defaults to models/<env_name>_policy.onnx",
)
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")


def get_rl_config(env_name: str):
    """Get RL config for the environment."""
    if env_name in envs.ALL_ENV_NAMES:
        return env_params.brax_ppo_config(env_name, _IMPL.value)
    elif env_name in mujoco_playground.manipulation._envs:
        return manipulation_params.brax_ppo_config(env_name, _IMPL.value)
    elif env_name in mujoco_playground.locomotion._envs:
        return locomotion_params.brax_ppo_config(env_name, _IMPL.value)
    elif env_name in mujoco_playground.dm_control_suite._envs:
        return dm_control_suite_params.brax_ppo_config(env_name, _IMPL.value)
    raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")


class MLP(tf.keras.Model):
    """Multi-layer perceptron for policy network."""

    def __init__(
        self,
        layer_sizes,
        activation=tf.nn.relu,
        kernel_init="lecun_uniform",
        activate_final=False,
        bias=True,
        layer_norm=False,
        mean_std=None,
    ):
        super().__init__()

        self.layer_sizes = layer_sizes
        self.activation = activation
        self.kernel_init = kernel_init
        self.activate_final = activate_final
        self.bias = bias
        self.layer_norm = layer_norm

        if mean_std is not None:
            self.mean = tf.Variable(mean_std[0], trainable=False, dtype=tf.float32)
            self.std = tf.Variable(mean_std[1], trainable=False, dtype=tf.float32)
        else:
            self.mean = None
            self.std = None

        self.mlp_block = tf.keras.Sequential(name="MLP_0")
        for i, size in enumerate(self.layer_sizes):
            dense_layer = layers.Dense(
                size,
                activation=self.activation,
                kernel_initializer=self.kernel_init,
                name=f"hidden_{i}",
                use_bias=self.bias,
            )
            self.mlp_block.add(dense_layer)
            if self.layer_norm:
                self.mlp_block.add(layers.LayerNormalization(name=f"layer_norm_{i}"))
        if not self.activate_final and self.mlp_block.layers:
            if (
                hasattr(self.mlp_block.layers[-1], "activation")
                and self.mlp_block.layers[-1].activation is not None
            ):
                self.mlp_block.layers[-1].activation = None

        self.submodules = [self.mlp_block]

    def call(self, inputs):
        if isinstance(inputs, list):
            inputs = inputs[0]
        if self.mean is not None and self.std is not None:
            inputs = (inputs - self.mean) / self.std
        logits = self.mlp_block(inputs)
        loc, _ = tf.split(logits, 2, axis=-1)
        return tf.tanh(loc)


def make_policy_network(
    param_size,
    mean_std,
    hidden_layer_sizes=(256, 256),
    activation=tf.nn.relu,
    kernel_init="lecun_uniform",
    layer_norm=False,
):
    """Create a TensorFlow policy network."""
    policy_network = MLP(
        layer_sizes=list(hidden_layer_sizes) + [param_size],
        activation=activation,
        kernel_init=kernel_init,
        layer_norm=layer_norm,
        mean_std=mean_std,
    )
    return policy_network


def transfer_weights(jax_params, tf_model):
    """
    Transfer weights from a JAX parameter dictionary to the TensorFlow model.

    Parameters:
    - jax_params: dict
      Nested dictionary with structure {block_name: {layer_name: {params}}}.
      For example:
      {
        'CNN_0': {
          'Conv_0': {'kernel': np.ndarray},
          'Conv_1': {'kernel': np.ndarray},
          'Conv_2': {'kernel': np.ndarray},
        },
        'MLP_0': {
          'hidden_0': {'kernel': np.ndarray, 'bias': np.ndarray},
          'hidden_1': {'kernel': np.ndarray, 'bias': np.ndarray},
          'hidden_2': {'kernel': np.ndarray, 'bias': np.ndarray},
        }
      }

    - tf_model: tf.keras.Model
      An instance of the adapted VisionMLP model containing named submodules and layers.
    """
    for layer_name, layer_params in jax_params.items():
        try:
            tf_layer = tf_model.get_layer("MLP_0").get_layer(name=layer_name)
        except ValueError:
            print(f"Layer {layer_name} not found in TensorFlow model.")
            continue
        if isinstance(tf_layer, tf.keras.layers.Dense):
            kernel = np.array(layer_params["kernel"])
            bias = np.array(layer_params["bias"])
            print(
                f"Transferring Dense layer {layer_name}, "
                f"kernel shape {kernel.shape}, bias shape {bias.shape}"
            )
            tf_layer.set_weights([kernel, bias])
        else:
            print(f"Unhandled layer type in {layer_name}: {type(tf_layer)}")

    print("Weights transferred successfully.")


def main(argv):
    del argv

    # Validate required flags
    if _CHECKPOINT_PATH.value is None:
        raise ValueError("--checkpoint_path is required")
    if _ENV_NAME.value is None:
        raise ValueError("--env_name is required")

    # Resolve checkpoint path
    ckpt_path = epath.Path(_CHECKPOINT_PATH.value).resolve()

    # Check if the path is a specific checkpoint (numeric directory name)
    if ckpt_path.is_dir() and ckpt_path.name.isdigit():
        # Direct checkpoint path, use as-is
        print(f"Using specified checkpoint: {ckpt_path}")
    else:
        # If the path is an experiment directory (contains 'checkpoints' subfolder), use that
        if (ckpt_path / "checkpoints").is_dir():
            ckpt_path = ckpt_path / "checkpoints"

        # Find the latest checkpoint in the directory
        if ckpt_path.is_dir():
            latest_ckpts = [
                ckpt
                for ckpt in ckpt_path.glob("*")
                if ckpt.is_dir() and ckpt.name.isdigit()
            ]
            if not latest_ckpts:
                raise ValueError(
                    f"No valid checkpoint directories found in {ckpt_path}. "
                    "Checkpoint directories should have numeric names."
                )
            latest_ckpts.sort(key=lambda x: int(x.name))
            ckpt_path = latest_ckpts[-1]
            print(f"Auto-selected latest checkpoint: {ckpt_path}")

    print(f"Loading checkpoint from: {ckpt_path}")

    # Determine output path
    if _OUTPUT_PATH.value is not None:
        output_path = Path(_OUTPUT_PATH.value)
    else:
        output_path = _ONNX_DIR / f"{_ENV_NAME.value}_policy.onnx"

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Output ONNX model will be saved to: {output_path}")

    # Get environment and config
    env_name = _ENV_NAME.value
    ppo_params = get_rl_config(env_name)
    env_cfg = registry.get_default_config(env_name)
    env_cfg["impl"] = _IMPL.value
    env = registry.load(env_name, config=env_cfg)

    obs_size = env.observation_size
    act_size = env.action_size
    print(f"Observation size: {obs_size}, Action size: {act_size}")

    # Create JAX network
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params.network_factory,
        # We need to explicitly call the normalization function here since only the brax
        # PPO train.py script creates it if normalize_observations is True.
        preprocess_observations_fn=running_statistics.normalize,
    )
    ppo_network = network_factory(obs_size, act_size)

    # Load checkpoint
    params = load(str(ckpt_path))
    params = (params[0], params[1])

    # Create JAX inference function for verification
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    inference_fn = make_inference_fn(params, deterministic=True)

    # Get observation key (usually 'state')
    obs_key = "state"
    if isinstance(obs_size, dict) and obs_key in obs_size:
        state_obs_size = obs_size[obs_key][0]
    else:
        state_obs_size = obs_size if isinstance(obs_size, int) else obs_size[0]

    # Extract mean/std for normalization
    mean = params[0].mean[obs_key]
    std = params[0].std[obs_key]
    # Convert mean/std jax arrays to tf tensors.
    mean_std = (tf.convert_to_tensor(mean), tf.convert_to_tensor(std))

    # Create TensorFlow policy network
    tf_policy_network = make_policy_network(
        param_size=act_size * 2,
        mean_std=mean_std,
        hidden_layer_sizes=ppo_params.network_factory.policy_hidden_layer_sizes,
        activation=tf.nn.swish,
    )

    # Build the model
    example_input = tf.zeros((1, state_obs_size))
    example_output = tf_policy_network(example_input)[0]
    print(f"TensorFlow model output shape: {example_output.shape}")

    # Transfer weights from JAX to TensorFlow
    transfer_weights(params[1]["params"], tf_policy_network)

    # Convert to ONNX
    spec = [tf.TensorSpec(shape=(1, state_obs_size), dtype=tf.float32, name="obs")]
    tf_policy_network.output_names = ["continuous_actions"]

    print("Converting to ONNX...")
    # opset 11 matches isaac lab.
    model_proto, _ = tf2onnx.convert.from_keras(
        tf_policy_network,
        input_signature=spec,
        opset=11,
        output_path=str(output_path),
    )
    print(f"ONNX model saved to: {output_path}")

    # Verify ONNX model
    print("\nVerifying ONNX model...")
    # Run inference with ONNX Runtime
    output_names = ["continuous_actions"]
    providers = ["CPUExecutionProvider"]
    onnx_session = rt.InferenceSession(str(output_path), providers=providers)

    # Test with ones
    test_input = np.ones((1, state_obs_size), dtype=np.float32)
    onnx_input = {"obs": test_input}
    onnx_pred = onnx_session.run(output_names, onnx_input)[0][0]

    # Compare with JAX prediction
    if isinstance(obs_size, dict):
        jax_test_input = {
            obs_key: jp.ones(obs_size[obs_key]),
        }
        # Add other observation keys with zeros
        for key in obs_size:
            if key != obs_key:
                jax_test_input[key] = jp.zeros(obs_size[key])
    else:
        jax_test_input = jp.ones((state_obs_size,))

    jax_pred, _ = inference_fn(jax_test_input, jax.random.PRNGKey(0))

    print(f"ONNX prediction: {onnx_pred[:5]}...")
    print(f"JAX prediction:  {np.array(jax_pred)[:5]}...")

    # Check if predictions match
    max_diff = np.max(np.abs(onnx_pred - np.array(jax_pred)))
    print(f"Max difference between ONNX and JAX: {max_diff:.6f}")

    if max_diff < 1e-4:
        print("✓ ONNX model matches JAX model!")
    else:
        print("⚠ Warning: ONNX and JAX predictions differ significantly.")

    print(f"\nConversion complete! ONNX model saved to: {output_path}")

    # import matplotlib.pyplot as plt

    # print(onnx_pred.shape)
    # print(example_output.shape)
    # print(jax_pred.shape)
    # plt.plot(onnx_pred, label="onnx")
    # plt.plot(example_output, label="tensorflow")
    # plt.plot(jax_pred, label="jax")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    flags.mark_flags_as_required(["checkpoint_path", "env_name"])
    app.run(main)
