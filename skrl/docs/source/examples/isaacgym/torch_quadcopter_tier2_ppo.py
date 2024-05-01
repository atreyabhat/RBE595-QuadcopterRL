import isaacgym

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaacgym_env_preview4
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed

# define shared model (stochastic and deterministic models) using mixins
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        # Define the CNN for depth image preprocessing
        self.cnn =  nn.Sequential(
                    nn.Conv2d(1, 8, kernel_size=10, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(8, 16, kernel_size=6, stride=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(800, 64),
                    nn.ReLU())


        self.quad_net =  nn.Sequential(
                            nn.Linear(13, 256),
                            nn.ELU(),
                            nn.Linear(256, 128),
                            nn.ELU(),
                            nn.Linear(128, 64),
                            nn.ELU(),
                            nn.Linear(64, 32),
                            nn.ELU())

        # Final layers
        self.final_net = nn.Sequential(
                        nn.Linear(64 + 32, 64),
                        nn.ELU(),
                        nn.Linear(64, 32),
                        nn.ELU())

        self.mean_layer = nn.Linear(32, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(32, 1)

    def act(self, inputs, role):
        # Preprocess the depth image with the CNN

        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):

        depth_image = inputs["states"][:, 13:]
        cnn_input = depth_image.view(-1, 1, 32, 32)

        cnn_output = self.cnn(cnn_input)
        quad_output = self.quad_net(inputs["states"][:, :13])
        final_output = self.final_net(torch.cat((cnn_output,quad_output), dim=1))

        if role == "policy":
            return self.mean_layer(final_output), self.log_std_parameter, {}
        elif role == "value":
            return self.value_layer(final_output), {}


# load and wrap the Isaac Gym environment
env = load_isaacgym_env_preview4(task_name="QuadcopterTier2")
env = wrap_env(env)

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=8, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Shared(env.observation_space, env.action_space, device)
models["value"] = models["policy"]  # same instance: shared model


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 8  # memory_size
cfg["learning_epochs"] = 8
cfg["mini_batches"] = 4  # 8 * 8192 / 16384
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 0.001
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.016}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 1.0
cfg["kl_threshold"] = 0
cfg["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.1
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 500
cfg["experiment"]["checkpoint_interval"] = 10000
cfg["experiment"]["directory"] = "runs/torch/Quadcopter_Tier2_PPO"

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 400000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# agent.load("runs/torch/Quadcopter_Tier2_PPO/24-04-29_21-39-56-742226_PPO/checkpoints/agent_30000.pt")


# start training
trainer.train()



# start evaluation
# trainer.eval()
