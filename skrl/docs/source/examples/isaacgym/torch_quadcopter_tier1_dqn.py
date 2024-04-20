import isaacgym

import gymnasium as gym

# import the skrl components to build the RL system
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaacgym_env_preview4

from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.model_instantiators.torch import Shape, deterministic_model


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


# load and wrap the Isaac Gym environment
env = load_isaacgym_env_preview4(task_name="QuadcopterTier1")
env = wrap_env(env)

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=32, num_envs=env.num_envs, device=device)

models = {}
models["q_network"] = deterministic_model(observation_space=env.observation_space,
                                          action_space=env.action_space,
                                          device=device,
                                          clip_actions=False,
                                          input_shape=Shape.OBSERVATIONS,
                                          hiddens=[256, 256],
                                          hidden_activation=["relu", "relu"],
                                          output_shape=Shape.ACTIONS,
                                          output_activation=None,
                                          output_scale=1.0)
models["target_q_network"] = deterministic_model(observation_space=env.observation_space,
                                                 action_space=env.action_space,
                                                 device=device,
                                                 clip_actions=False,
                                                 input_shape=Shape.OBSERVATIONS,
                                                 hiddens=[256, 256],
                                                 hidden_activation=["relu", "relu"],
                                                 output_shape=Shape.ACTIONS,
                                                 output_activation=None,
                                                 output_scale=1.0)

# initialize models' parameters (weights and biases)
for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/dqn.html#configuration-and-hyperparameters
cfg = DQN_DEFAULT_CONFIG.copy()
cfg["learning_starts"] = 100
cfg["exploration"]["final_epsilon"] = 0.04
cfg["exploration"]["timesteps"] = 1500
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 1000
cfg["experiment"]["checkpoint_interval"] = 5000
cfg["experiment"]["directory"] = "runs/torch/QuadcopterTier1_DQN"

agent = DQN(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 50000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# start training
trainer.train()