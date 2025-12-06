import gymnasium as gym
import numpy as np
import torch

import Env


class PPOAgent(torch.nn.Module):
    def __init__(self, observation_space_dim, action_space_dim, hidden_layer_dim):
        super(PPOAgent, self).__init__()

        def init_layer(layer: torch.nn.Linear) -> torch.nn.Linear:
            """
            Initializes the weight and bias of the neural network layer.
            """
            torch.nn.init.orthogonal_(layer.weight, np.sqrt(2))
            torch.nn.init.constant_(layer.bias, 0.0)
            return layer

        # Actor / Policy Network
        self.actor = torch.nn.Sequential(
            init_layer(torch.nn.Linear(in_features=observation_space_dim, out_features=hidden_layer_dim, dtype=torch.float32)),
            torch.nn.Tanh(),
            init_layer(torch.nn.Linear(in_features=hidden_layer_dim, out_features=hidden_layer_dim, dtype=torch.float32)),
            torch.nn.Tanh(),
            init_layer(torch.nn.Linear(in_features=hidden_layer_dim, out_features=action_space_dim, dtype=torch.float32)),
        )

        # Critic / Value Network
        self.critic = torch.nn.Sequential(
            init_layer(torch.nn.Linear(in_features=observation_space_dim, out_features=hidden_layer_dim, dtype=torch.float32)),
            torch.nn.Tanh(),
            init_layer(torch.nn.Linear(in_features=hidden_layer_dim, out_features=hidden_layer_dim, dtype=torch.float32)),
            torch.nn.Tanh(),
            init_layer(torch.nn.Linear(in_features=hidden_layer_dim, out_features=1, dtype=torch.float32))
        )


    def get_action_logits(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Returns a probability distribution of actions given the observation.
        """
        return self.actor.__call__(observation)
    

    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Returns a value given the observation.
        """
        return self.critic.__call__(observation)
    

# Design related parameters 
DESIGNS_ROOT_DIR = "./2-designs"

# HPC related parameters
JOB_SCRIPTS_ROOT_DIR = "./3-job-scripts"


def train_ppo_agent(kernel_info: str, ppo_agent_model_hidden_layer_dim: int, ppo_agent_model_path: str):
    """
    Trains a PPO agent.

    Args:
        kernel_info (dict):                     The kernel info including number of resources utilized, etc.
        ppo_agent_model_hidden_layer_dim (int): The number of hidden layers of the neural network of the PPO agent.
        ppo_agent_model_path (str):             The path to the neural network of the PPO agent. (e.g., "./ppo-agent.pth")
    """

    # Gym related parameter
    NUM_ENVS = 16
    
    # PPO related parameters
    NUM_ITERATIONS = 10
    NUM_STEPS = 128
    NUM_EPOCHS = 4

    BATCH_SIZE = NUM_STEPS * NUM_ENVS
    MINIBATCH_SIZE = 64

    GAMMA = 0.99
    LAMBDA = 0.95
    EPSILON_POLICY = 0.2
    EPSILON_VALUE  = 0.5

    # Create an environment
    envs = gym.vector.AsyncVectorEnv([
        lambda: gym.make(id="yiyang/Env-v0", env_idx=1,  designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=2,  designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=3,  designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=4,  designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=5,  designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=6,  designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=7,  designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=8,  designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=9,  designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=10, designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=11, designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=12, designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=13, designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=14, designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=15, designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=16, designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
    ])

    # Flatten the observations for convenience
    envs = gym.wrappers.vector.FlattenObservation(envs)

    num_actions = gym.spaces.flatdim(envs.single_action_space)
    num_observations = gym.spaces.flatdim(envs.single_observation_space)

    # Create an agent and optimizer
    agent = PPOAgent(observation_space_dim=num_observations, action_space_dim=num_actions, hidden_layer_dim=ppo_agent_model_hidden_layer_dim)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-5, eps=1e-5)

    torch.manual_seed(seed=0)

    # shape (NUM_STEPS, NUM_ENVS, NUM_OBS)
    if envs.single_observation_space.shape is not None:
        total_batched_observations: torch.Tensor = torch.full(size=(NUM_STEPS, NUM_ENVS) + envs.single_observation_space.shape, fill_value=0.0, dtype=torch.float32)
    else:
        total_batched_observations: torch.Tensor = torch.full(size=(NUM_STEPS, NUM_ENVS), fill_value=0.0, dtype=torch.float32)
    # print(f"Total Batched Observations: {total_batched_observations} (shape: {total_batched_observations.shape}, dtype: {total_batched_observations.dtype})")

    # shape (NUM_STEPS, NUM_ENVS,)
    if envs.single_action_space.shape is not None:
        total_batched_actions: torch.Tensor = torch.full(size=(NUM_STEPS, NUM_ENVS) + envs.single_action_space.shape, fill_value=0, dtype=torch.int64)
    else:
        total_batched_actions: torch.Tensor = torch.full(size=(NUM_STEPS, NUM_ENVS), fill_value=0, dtype=torch.int64)
    # print(f"Total Batched Actions: {total_batched_actions} (shape: {total_batched_actions.shape}, dtype: {total_batched_actions.dtype})")
    
    total_batched_action_log_probs: torch.Tensor = torch.full(size=(NUM_STEPS, NUM_ENVS), fill_value=0.0,   dtype=torch.float32)
    total_batched_rewards: torch.Tensor          = torch.full(size=(NUM_STEPS, NUM_ENVS), fill_value=0.0,   dtype=torch.float32)
    total_batched_is_done: torch.Tensor          = torch.full(size=(NUM_STEPS, NUM_ENVS), fill_value=False, dtype=bool)
    total_batched_values: torch.Tensor           = torch.full(size=(NUM_STEPS, NUM_ENVS), fill_value=0.0,   dtype=torch.float32)
    total_batched_advantages: torch.Tensor       = torch.full(size=(NUM_STEPS, NUM_ENVS), fill_value=0.0,   dtype=torch.float32)
    total_batched_returns: torch.Tensor          = torch.full(size=(NUM_STEPS, NUM_ENVS), fill_value=0.0,   dtype=torch.float32)

    # shape (NUM_ENVS, DIM_OBSERVATION_SPACE), (NUM_ENVS, DIM_INFO)
    batched_observations, batched_info = envs.reset(seed=0)
    # print(f"Batched Observations: {batched_observations} (shape: {batched_observations.shape})")

    # shape (NUM_ENVS, DIM_OBSERVATION_SPACE)
    batched_observations: torch.Tensor = torch.tensor(batched_observations, dtype=torch.float32)
    # print(f"Batched Observations: {batched_observations} (shape: {batched_observations.shape}, dtype: {batched_observations.dtype})")

    # shape (NUM_ENVS, NUM_LEGAL_DIRECTIVES)
    # TODO: uncomment below line to enable masked action
    # batched_action_masks: torch.Tensor = torch.tensor(batched_info["legal_directive_idx_mask"], dtype=torch.float32)
    # print(f"Batched Action Masks: {batched_action_masks} (shape: {batched_action_masks.shape}, dtype: {batched_action_masks.dtype})")

    for iteration in range(NUM_ITERATIONS):

        for step in range(NUM_STEPS):

            # During each STEP, execute and record an action
            with torch.no_grad():
                # shape (NUM_ENVS, NUM_ACTIONS)
                batched_action_logits: torch.Tensor = agent.get_action_logits(batched_observations)
                # print(f"Batched Action Logits: {batched_action_logits} (shape: {batched_action_logits.shape})")

                # shape (NUM_ENVS, NUM_ACTIONS)
                # TODO: uncomment below line to enable masked action
                # batched_masked_action_logits: torch.Tensor = batched_action_logits + (1 - batched_action_masks) * -1e9
                # print(f"Batched Masked Action Logits: {batched_masked_action_logits} (shape: {batched_masked_action_logits.shape})")

                # shape (NUM_ENVS, NUM_ACTIONS)
                batched_action_prob_dist = torch.distributions.categorical.Categorical(logits=batched_action_logits)
                # TODO: uncomment below line to enable masked action
                # batched_action_prob_dist = torch.distributions.categorical.Categorical(logits=batched_masked_action_logits)
                # print(f"Batched Action Prob Dist: {batched_action_prob_dist}")

                # shape (NUM_ENVS,)
                batched_actions: torch.Tensor = batched_action_prob_dist.sample()
                # print(f"Batched Action Indices: {batched_action_indices} (shape: {batched_action_indices.shape}, dtype: {batched_action_indices.dtype})")

                # shape (NUM_ENVS,)
                batched_action_log_probs: torch.Tensor = batched_action_prob_dist.log_prob(batched_actions)
                # print(f"Batched Action Log Probs: {batched_action_log_probs} (shape: {batched_action_log_probs.shape}, dtype: {batched_action_log_probs.dtype})")

                # shape (NUM_ENVS, 1)
                batched_values: torch.Tensor = agent.get_value(batched_observations)
                # print(f"Batched Values: {batched_values} (shape: {batched_values.shape}, dtype: {batched_values.dtype})")

                # shape (NUM_ENVS,)
                batched_values: torch.Tensor = torch.flatten(batched_values)
                # print(f"Batched Values: {batched_values} (shape: {batched_values.shape}, dtype: {batched_values.dtype})")

                # Record
                total_batched_actions[step] = batched_actions
                total_batched_action_log_probs[step] = batched_action_log_probs
                total_batched_values[step] = batched_values
            
            # Execute the chose actions
            batched_observations, batched_rewards, batched_is_terminated, batched_is_truncated, _, = envs.step(batched_actions)

            # shape (NUM_ENVS, NUM_OBS)
            batched_observations: torch.Tensor = torch.tensor(batched_observations, dtype=torch.float32)
            # print(f"Batched Observations: {batched_observations} (shape: {batched_observations.shape}, dtype: {batched_observations.dtype})")

            # shape (NUM_ENVS,)
            batched_rewards: torch.Tensor = torch.tensor(batched_rewards, dtype=torch.float32)
            # print(f"Batched Rewards: {batched_rewards} (shape: {batched_rewards.shape}, dtype: {batched_rewards.dtype})")

            # shape (NUM_ENVS,)
            batched_is_done: torch.Tensor = torch.logical_or(torch.tensor(batched_is_terminated), torch.tensor(batched_is_truncated))
            # print(f"Batched Is Done: {batched_is_done} (shape: {batched_is_done.shape}, dtype: {batched_is_done.dtype})")

            # Record
            total_batched_rewards[step] = batched_rewards


        # During each ITERATION, calculate the returns
        with torch.no_grad():
            # shape (NUM_ENVS, 1)
            batched_values: torch.Tensor = agent.get_value(batched_observations)
            # print(f"Batched Values: {batched_values} (shape: {batched_values.shape}, dtype: {batched_values.dtype})")

            # shape (NUM_ENVS,)
            batched_values: torch.Tensor = torch.flatten(batched_values)
            # print(f"Batched Values: {batched_values} (shape: {batched_values.shape}, dtype: {batched_values.dtype})")

            batched_advantages = 0

            for i in reversed(range(NUM_STEPS)):
                if i == NUM_STEPS - 1:
                    # shape (NUM_ENVS,)
                    batched_is_not_done: torch.Tensor = torch.logical_not(batched_is_done)
                    # print(f"Batched Is Not Done: {batched_is_not_done} (shape: {batched_is_not_done.shape}, dtype: {batched_is_not_done.dtype})")

                    # shape (NUM_ENVS,)
                    batched_deltas: torch.Tensor = total_batched_rewards[i] + GAMMA * batched_values * batched_is_not_done - total_batched_values[i]
                    # print(f"Batched Deltas: {batched_deltas} (shape: {batched_deltas.shape}, dtype: {batched_deltas.dtype})")

                    # shape (NUM_ENVS,)
                    batched_advantages: torch.Tensor = batched_deltas + GAMMA * LAMBDA * batched_is_not_done * batched_advantages
                    # print(f"Batched Advantages: {batched_advantages} (shape: {batched_advantages.shape}, dtype: {batched_advantages.dtype})")
                else:
                    batched_is_not_done: torch.Tensor = torch.logical_not(total_batched_is_done[i + 1])
                    batched_deltas: torch.Tensor = total_batched_rewards[i] + GAMMA * total_batched_values[i + 1] * batched_is_not_done - total_batched_values[i]
                    batched_advantages: torch.Tensor = batched_deltas + GAMMA * LAMBDA * batched_is_not_done * batched_advantages
                
                total_batched_advantages[i] = batched_advantages

            total_batched_returns: torch.Tensor = total_batched_advantages + total_batched_values


        # shape (NUM_STEPS * NUM_ENVS, NUM_OBS)
        if envs.single_observation_space.shape is not None:
            total_flatten_batched_observations: torch.Tensor = total_batched_observations.reshape((-1,) + envs.single_observation_space.shape)
        else:
            total_flatten_batched_observations: torch.Tensor = total_batched_observations.reshape(-1)
        # print(f"Total Batched Observations: (shape: {total_batched_observations.shape}, dtype: {total_batched_observations.dtype})")

        # shape (NUM_STEPS * NUM_ENVS,)
        if envs.single_action_space.shape is not None:
            total_flatten_batched_action_indices: torch.Tensor = total_batched_actions.reshape((-1,) + envs.single_action_space.shape)
        else:
            total_flatten_batched_action_indices: torch.Tensor = total_batched_actions.reshape(-1)
        # print(f"Total Batched Action Indices: (shape: {total_batched_action_indices.shape}, dtype: {total_batched_action_indices.dtype})")

        # shape (NUM_STEPS * NUM_ENVS,)
        total_flatten_batched_action_log_probs: torch.Tensor = total_batched_action_log_probs.reshape(-1)
        # print(f"Total Batched Action Log Probs: (shape: {total_batched_action_log_probs.shape}, dtype: {total_batched_action_log_probs.dtype})")

        # shape (NUM_STEPS * NUM_ENVS,)
        total_flatten_batched_values: torch.Tensor = total_batched_values.reshape(-1)
        # print(f"Total Batched Values: (shape: {total_batched_values.shape}, dtype: {total_batched_values.dtype})")

        # shape (NUM_STEPS * NUM_ENVS,)
        total_flatten_batched_advantages: torch.Tensor = total_batched_advantages.reshape(-1)
        # print(f"Total Batched Advantages: (shape: {total_batched_advantages.shape}, dtype: {total_batched_advantages.dtype})")

        # shape (NUM_STEPS * NUM_ENVS,)
        total_flatten_batched_returns: torch.Tensor = total_batched_returns.reshape(-1)
        # print(f"Total Batched Returns: (shape: {total_batched_returns.shape}, dtype: {total_batched_returns.dtype})")

        # During each ITERATION, optimize the policy and value network
        for epoch in range(NUM_EPOCHS):
            batch_indices: torch.Tensor = torch.randperm(BATCH_SIZE)

            for i in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                minibatch_indices: torch.Tensor = batch_indices[i : i + MINIBATCH_SIZE]

                # shape (MINIBATCH_SIZE, NUM_ACTIONS)
                minibatched_action_logits: torch.Tensor = agent.get_action_logits(total_flatten_batched_observations[minibatch_indices])
                minibatched_action_prob_dist = torch.distributions.categorical.Categorical(logits=minibatched_action_logits)
                minibatched_action_log_probs: torch.Tensor = minibatched_action_prob_dist.log_prob(total_flatten_batched_action_indices[minibatch_indices])
                # print(f"Mini-Batched Action Log Probs: (shape: {minibatched_action_log_probs.shape}, dtype: {minibatched_action_log_probs.dtype})")

                # shape (MINIBATCH_SIZE,)
                minibatched_action_log_probs_diff: torch.Tensor = minibatched_action_log_probs - total_flatten_batched_action_log_probs[minibatch_indices]
                minibatched_policy_ratio: torch.Tensor = torch.exp(minibatched_action_log_probs_diff)
                # print(f"Mini-Batched Policy Ratio: (shape: {minibatched_policy_ratio.shape}, dtype: {minibatched_policy_ratio.dtype})")

                # shape (MINIBATCH_SIZE,)
                minibatched_advantages: torch.Tensor = total_flatten_batched_advantages[minibatch_indices]
                minibatched_advantages: torch.Tensor = (minibatched_advantages - torch.mean(minibatched_advantages)) / (torch.std(minibatched_advantages) + 1e-8)
                # print(f"Mini-Batched Advantages: (shape: {minibatched_advantages.shape}, dtype: {minibatched_advantages.dtype})")
                
                # shape (MINIBATCH_SIZE,)
                policy_loss_unclipped: torch.Tensor = minibatched_advantages * minibatched_policy_ratio
                policy_loss_clipped: torch.Tensor = minibatched_advantages * torch.clamp(minibatched_policy_ratio, 1 - EPSILON_POLICY, 1 + EPSILON_POLICY)
                # print(f"Policy Loss Clipped: (shape: {policy_loss_clipped.shape}, dtype: {policy_loss_clipped.dtype})")

                # shape (MINIBATCH_SIZE, 1)
                minibatched_values: torch.Tensor = agent.get_value(total_flatten_batched_observations[minibatch_indices])
                # print(f"Mini-Batched Values: (shape: {minibatched_values.shape}, dtype: {minibatched_values.dtype})")

                # shape (MINIBATCH_SIZE,)
                minibatched_values: torch.Tensor = torch.flatten(minibatched_values)
                # print(f"Mini-Batched Values: (shape: {minibatched_values.shape}, dtype: {minibatched_values.dtype})")
                
                # shape (MINIBATCH_SIZE,)
                value_loss_unclipped: torch.Tensor = (minibatched_values - total_flatten_batched_returns[minibatch_indices])**2
                value_loss_clipped: torch.Tensor = (total_flatten_batched_values[minibatch_indices] + torch.clamp(minibatched_values - total_flatten_batched_values[minibatch_indices], -EPSILON_POLICY, EPSILON_POLICY) - total_flatten_batched_returns[minibatch_indices])**2
                # print(f"Value Loss Clipped: (shape: {value_loss_clipped.shape}, dtype: {value_loss_clipped.dtype})")

                policy_loss: torch.float32 = torch.mean(-torch.min(policy_loss_unclipped, policy_loss_clipped))
                value_loss: torch.float32 = 0.5 * torch.mean(torch.max(value_loss_unclipped, value_loss_clipped)) * EPSILON_VALUE
                total_loss: torch.float32 = policy_loss + value_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

            if epoch == NUM_EPOCHS - 1:
                print(f"Iteration: #{iteration}, Loss: {total_loss.item()}")

    torch.save(agent.state_dict(), ppo_agent_model_path)
    envs.close()


def evaluate_ppo_agent(kernel_info: str, ppo_agent_model_hidden_layer_dim: int, ppo_agent_model_path: str):
    """
    Evaluates the performance of a PPO agent.

    Args:
        kernel_info (dict):                     The kernel info including number of resources utilized etc.
        ppo_agent_model_hidden_layer_dim (int): The number of hidden layers of the neural network of the PPO agent.
        ppo_agent_model_path (str):             The path to the neural network of the PPO agent. (e.g., "./ppo-agent.pth")
    """

    # Create environments
    envs = gym.vector.AsyncVectorEnv([
        lambda: gym.make(id="yiyang/Env-v0", env_idx=1,  designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=2,  designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=3,  designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=4,  designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=5,  designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=6,  designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=7,  designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=8,  designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=9,  designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=10, designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=11, designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=12, designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=13, designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=14, designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=15, designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
        lambda: gym.make(id="yiyang/Env-v0", env_idx=16, designs_root_dir=DESIGNS_ROOT_DIR, job_scripts_root_dir=JOB_SCRIPTS_ROOT_DIR, kernel_info=kernel_info),
    ])

    # Flatten the observations for convenience
    envs = gym.wrappers.vector.FlattenObservation(envs)

    num_actions = gym.spaces.flatdim(envs.single_action_space)
    num_observations = gym.spaces.flatdim(envs.single_observation_space)

    # Create an agent and load its parameters
    agent = PPOAgent(observation_space_dim=num_observations, action_space_dim=num_actions, hidden_layer_dim=ppo_agent_model_hidden_layer_dim)
    agent.load_state_dict(torch.load(ppo_agent_model_path))
    agent.eval()

    # Evaluate the agent
    for i in range(1):
        batched_observations, batched_info = envs.reset(seed=0)
    
        while True:
            # shape (NUM_ENVS, DIM_OBSERVATION_SPACE)
            batched_observations: torch.Tensor = torch.tensor(batched_observations, dtype=torch.float32)

            # shape (NUM_ENVS, NUM_LEGAL_DIRECTIVES)
            # TODO: uncomment below line to enable masked action
            # batched_action_masks: torch.Tensor = torch.tensor(batched_info["legal_directive_idx_mask"], dtype=torch.float32)

            # Obtain action index
            with torch.no_grad():
                # shape (NUM_ENVS, NUM_ACTIONS)
                batched_action_logits: torch.Tensor = agent.get_action_logits(batched_observations)
                # print(f"Batched Action Logits: {action_logits} (shape: {action_logits.shape})")

                # shape (NUM_ENVS, NUM_ACTIONS)
                # TODO: uncomment below line to enable masked actions
                # batched_masked_action_logits: torch.Tensor = batched_action_logits + (1 - batched_action_masks) * -1e9

                # shape (NUM_ENVS, NUM_ACTIONS)
                batched_action_prob_dist = torch.distributions.categorical.Categorical(logits=batched_action_logits)
                # TODO: uncomment below line to enable masked actions
                # batched_action_prob_dist = torch.distributions.categorical.Categorical(logits=batched_masked_action_logits)
                batched_actions: torch.Tensor = batched_action_prob_dist.sample()

            batched_observations, batched_rewards, batched_is_terminated, batched_is_truncated, _, = envs.step(batched_actions)
            
            if np.any(batched_is_terminated):
                print(f"Episode {i+1} terminated with reward:\n {batched_rewards}")
                print(f"Observation:\n {batched_observations}, ({type(batched_observations)})")
                break
            
            if np.any(batched_is_truncated):
                print(f"Episode {i+1} truncated with reward:\n {batched_rewards}")
                print(f"Observation:\n {batched_observations}, ({type(batched_observations)})")
                break
            
            print(f"Reward:\n {batched_rewards}")
                
    envs.close()
    