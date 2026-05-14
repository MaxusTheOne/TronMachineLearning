from abc import ABC, abstractmethod
import numpy as np

class TronAgent(ABC):
    """Abstract agent for the Tron batch environment."""

    @property
    @abstractmethod
    def observation_type(self) -> str:
        """'lite' or 'grid' – tells the runner what observation to give this agent."""
        ...

    @abstractmethod
    def act(
        self,
        observation: np.ndarray,      # float, shape [envs, ...] depending on type
        legal_actions: np.ndarray,    # bool, [envs, 3]
    ) -> np.ndarray:
        """Return actions for all envs, shape [envs], int in {0,1,2}."""
        ...


# Author: Greyson Wintergerst
# Last Updated: 12/1/2024
# Purpose: This file contains the DeepQNetwork class and DeepQAgent class. The DeepQNetwork class is a neural network
#          that is used to approximate the Q-values of the agent in the game. This network uses a combination of convolutional
#          and dense layers to process the game board and player state inputs.
#          The DeepQAgent class is the agent that uses the DeepQNetwork to learn the optimal policy for the game. The agent
#          uses an epsilon-greedy policy to explore the environment and update the Q-values of the network. The agent also
#          uses experience replay to sample random batches of experiences and train the network on these samples.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import json


# Defines the neural network architecture
class DeepQNetwork(nn.Module):
    def __init__(self, map_shape=(40, 40), players_state_size=6, action_size=4):
        super(DeepQNetwork, self).__init__()

        # Convolutional Layers for the game board
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Flatten output of convolutional layers
        conv_output_size = map_shape[0] * map_shape[1] * 64

        # Fully connected layers for the game board
        self.fc_player = nn.Linear(players_state_size, 128)

        # Combined layers
        self.fc_combined1 = nn.Linear(conv_output_size + 128, 128)
        self.fc_combined2 = nn.Linear(128, 128)
        self.fc_output = nn.Linear(128, action_size)

    # Forward pass of the network
    def forward(self, map_input, players_state_input):
        # Convolutional pathway for the map
        x_map = F.relu(self.conv1(map_input))
        x_map = F.relu(self.conv2(x_map))
        x_map = F.relu(self.conv3(x_map))
        x_map = x_map.view(x_map.size(0), -1)  # Flatten

        # Dense pathway for the other state values
        x_player = F.relu(self.fc_player(players_state_input))

        # Combine the two pathways
        x = torch.cat((x_map, x_player), dim=1)
        x = F.relu(self.fc_combined1(x))
        x = F.relu(self.fc_combined2(x))
        x = self.fc_output(x)

        return x
class MLPPolicy:
    def __init__(self, obs_dim, hidden=32, genome=None, rng=None):
        self.obs_dim = int(obs_dim)
        self.hidden = int(hidden)
        self.rng = np.random.default_rng(rng)

        self.n_params = (
            self.obs_dim * self.hidden + self.hidden +
            self.hidden * 3 + 3
        )

        if genome is None:
            genome = self.rng.normal(0, 0.1, size=self.n_params).astype(np.float32)
        else:
            genome = np.asarray(genome, dtype=np.float32)
        self.genome = genome
        self._unpack_genome(genome)

    def _unpack_genome(self, genome):
        assert genome.shape == (self.n_params,)
        i = 0
        n = self.obs_dim * self.hidden
        self.w1 = genome[i:i+n].reshape(self.obs_dim, self.hidden)
        i += n

        self.b1 = genome[i:i+self.hidden]
        i += self.hidden

        n = self.hidden * 3
        self.w2 = genome[i:i+n].reshape(self.hidden, 3)
        i += n

        self.b2 = genome[i:i+3]

    def set_genome(self, genome):
        genome = np.asarray(genome, dtype=np.float32)
        self.genome = genome
        self._unpack_genome(genome)

    def get_genome(self):
        return self.genome

    def logits(self, observations):
        # obs: [envs, players, obs_dim]
        h = np.tanh(observations @ self.w1 + self.b1)
        return h @ self.w2 + self.b2

    def actions(self, model):
        observations = model.observe_lite()
        logits = self.logits(observations)
        return logits.argmax(axis=-1).astype(np.int8)


# Defines the agent that uses the DeepQNetwork to learn the optimal policy
class DeepQAgent(TronAgent):
    def __init__(self, gameObj=None, state_size=None, action_size=4, hidden_size=128, gamma=0.95, epsilon=1.0, 
                 epsilon_min=0.01, epsilon_decay=0.995, alpha=0.001, obs_type="grid", device="cpu", deterministic=False):
        self.gameObj = gameObj
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=5000)  # store maximum of 5000 states in "memory"
        self.alpha = alpha  # learning rate
        self._obs_type = obs_type  # 'lite' or 'grid'
        self.device = device
        self.deterministic = deterministic

        self.model = DeepQNetwork()
        self.target_model = DeepQNetwork()  # target network for fixed Q-targets
        self.target_model.load_state_dict(
            self.model.state_dict())  # initialize target model with the same weights as the model

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)  # optimizer for the model
        self.criterion = nn.MSELoss()  # loss function

        # used for storing training metrics
        self.training_metrics = {
            "episode": [],
            "total_reward": [],
            "average_loss": [],
            "epsilon": []
        }

    @property
    def observation_type(self) -> str:
        """Return the observation type this agent expects."""
        return self._obs_type

    # Updates the target model with the weights of the learning model
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, observation: np.ndarray, legal_actions: np.ndarray) -> np.ndarray:
        """
        TronBattleRunner-compatible act method.
        
        Args:
            observation: np.ndarray of shape [envs, ...] (depends on observation_type)
                - If 'grid': [envs, 1+players, height, width]
                - If 'lite': [envs, features]
            legal_actions: np.ndarray of shape [envs, 3] bool, indicating legal actions
        
        Returns:
            np.ndarray of shape [envs], int in {0, 1, 2}
        """
        batch_size = observation.shape[0]
        actions = np.zeros(batch_size, dtype=np.int8)
        
        for env_idx in range(batch_size):
            # Get legal actions for this env
            legal = legal_actions[env_idx]  # [3]
            
            if self.deterministic or np.random.rand() > self.epsilon:
                # Exploitation: use model to predict best action
                action = self._predict_action(observation[env_idx], legal)
            else:
                # Exploration: choose random legal action
                legal_indices = np.where(legal)[0]
                if len(legal_indices) > 0:
                    action = np.random.choice(legal_indices).astype(np.int8)
                else:
                    action = np.int8(0)
            
            actions[env_idx] = action
        
        return actions

    def act_single(self, state, playerid=2):
        """
        Legacy method for single-env training (e.g., in the training loop).
        Chooses an action based on the epsilon-greedy policy.
        
        Args:
            state: dict with 'map' and 'player' keys
            playerid: player ID
            
        Returns:
            int action in {0, 1, 2}
        """
        valid_actions = [
            action for action in range(self.action_size)
            if self.gameObj and not self.gameObj.players[playerid].isCollision(action)
        ] if self.gameObj else list(range(3))

        if np.random.rand() <= self.epsilon:  # explore
            return int(random.choice(valid_actions)) if valid_actions else 0

        # Exploitation
        return self._predict_action_single(state, valid_actions)

    def _predict_action(self, observation: np.ndarray, legal_actions: np.ndarray) -> np.int8:
        """
        Predict action for a single environment using the model.
        
        Args:
            observation: observation for one env
            legal_actions: [3] bool array of legal actions
            
        Returns:
            int action in {0, 1, 2}
        """
        if self._obs_type == "grid":
            # observation shape: [1+players, height, width]
            map_input = torch.tensor(observation[0:1], dtype=torch.float32).unsqueeze(0)  # [1, 1, h, w]
            # For now, use dummy player state; consider extracting from observation if available
            players_input = torch.zeros((1, 6), dtype=torch.float32)
        else:  # 'lite'
            # observation is flattened features [features]
            # Convert to dummy map and player state for compatibility
            map_input = torch.zeros((1, 1, 40, 40), dtype=torch.float32)
            players_input = torch.tensor(observation[:6], dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.model(map_input, players_input).squeeze()  # [3]
        
        # Mask illegal actions
        q_values[~torch.tensor(legal_actions, dtype=torch.bool)] = float('-inf')
        
        action = torch.argmax(q_values).item()
        return np.int8(action)

    # Returns the action with the highest Q-value (legacy single-env version)
    def _predict_action_single(self, state, valid_actions):
        """Legacy single-environment prediction."""
        map_input = torch.tensor(state["map"], dtype=torch.float32).unsqueeze(0).unsqueeze(
            0)  # add batch and channel dimensions
        players_input = torch.tensor(state["player"], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(map_input, players_input).squeeze()

        for action in range(self.action_size):
            if action not in valid_actions:
                q_values[action] = -10

        return int(torch.argmax(q_values).item())

    # Primary training function for the agent; replays saved memory and updates the model
    # Returns loss for the batch
    def replay(self, batch_size):
        """Train the model using experience replay."""
        if len(self.memory) < batch_size:
            return None

        # Sample a random batch of experiences
        memBatch = random.sample(self.memory, batch_size)

        # Prepare batched inputs
        map_batch = []
        players_batch = []
        targets = []

        for state, action, reward, next_state, done in memBatch:
            # Prepare current state inputs
            map_batch.append(torch.tensor(state["map"], dtype=torch.float32))
            players_batch.append(torch.tensor(state["player"], dtype=torch.float32))

            # Compute the target Q-value
            target = reward
            if not done:
                next_map_input = torch.tensor(next_state["map"], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                next_players_input = torch.tensor(next_state["player"], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    next_q_values = self.target_model(next_map_input, next_players_input)
                    target += self.gamma * torch.max(next_q_values).item()

            # Target tensor for the action taken
            targets.append((action, target))

        # Stack map and other inputs to form batched tensors
        map_batch = torch.stack(map_batch).unsqueeze(1)  # Shape: [batch_size, 1, 40, 40]
        players_batch = torch.stack(players_batch)  # Shape: [batch_size, 6]

        # Forward pass and compute loss for the batch
        current_q = self.model(map_batch, players_batch)
        target_q = current_q.clone().detach()

        for i, (action, target) in enumerate(targets):
            target_q[i][action] = target  # Update only the Q-value for the action taken

        self.optimizer.zero_grad()
        loss = self.criterion(current_q, target_q)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    # Store the current state, action, reward, next state, and done flag in memory
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory for later replay."""
        self.memory.append((state, action, reward, next_state, done))

    # Training loop that manages overall training process
    def train(self, env, episodes, batch_size, playerid):
        """Train the agent in the given environment (legacy single-env training)."""

        for e in range(episodes):
            if e % 10 == 0:
                self.update_target_model()

            state = env.reset(model=self)
            total_reward = 0
            done = False
            losses = []

            while not done:
                action = self.act_single(state, playerid)
                next_state, rewards, done = env.step()

                reward = rewards[playerid]
                self.remember(
                    {"map": state["map"], "player": state["player"]},
                    action,
                    reward,
                    {"map": next_state["map"], "player": next_state["player"]},
                    done
                )

                total_reward += reward

                # Replay experience
                loss = self.replay(batch_size)
                if loss is not None:
                    losses.append(loss)

            avg_loss = sum(losses) / len(losses) if losses else 0
            self.training_metrics["episode"].append(e)
            self.training_metrics["total_reward"].append(total_reward)
            self.training_metrics["average_loss"].append(avg_loss)
            self.training_metrics["epsilon"].append(self.epsilon)

            if e % 10 == 0:
                print(f"Episode {e}/{episodes}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.4f}")

        self.save("deepq_model.pth")
        self.save_training_metrics()
        print("Training complete. Model saved to deepq_model.pth. Training metrics saved to training_metrics.json")

    # Saves model weights after training
    def save(self, path):
        """Save the model weights to a file."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    # Loads model weights from a file given a file path
    def load(self, path):
        """Load model weights from a file."""
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")

    # Saves training metrics to a json file
    def save_training_metrics(self, filename="training_metrics.json"):
        """Save training metrics to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.training_metrics, f)
            print(f"Training metrics saved to {filename}")

    def set_observation_type(self, obs_type: str):
        """Set the observation type ('lite' or 'grid')."""
        if obs_type not in ['lite', 'grid']:
            raise ValueError(f"obs_type must be 'lite' or 'grid', got {obs_type}")
        self._obs_type = obs_type

    def set_deterministic(self, deterministic: bool):
        """Set whether to use deterministic (greedy) action selection."""
        self.deterministic = deterministic

    def set_epsilon(self, epsilon: float):
        """Set the epsilon value for epsilon-greedy exploration."""
        self.epsilon = max(self.epsilon_min, min(1.0, float(epsilon)))

