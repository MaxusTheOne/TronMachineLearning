"""
Genetic Algorithm trainer for Tron battle game.

Supports:
- Training against self (self-play)
- Training against GreedyController
- Checkpointing and continuing training
- Population management with elitism
- Genome serialization/deserialization
"""

from __future__ import annotations

import os
import pickle
import json
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Callable
from pathlib import Path

import numpy as np

from batch_model import TronBatchModel, StepResult


@dataclass
class Genome:
    """Neural network genome for Tron agent."""

    weights: List[np.ndarray]  # List of weight matrices
    biases: List[np.ndarray]  # List of bias vectors
    fitness: float = 0.0
    generation: int = 0

    def __post_init__(self):
        """Ensure consistent types."""
        self.weights = [w.astype(np.float32) for w in self.weights]
        self.biases = [b.astype(np.float32) for b in self.biases]

    def mutate(self, mutation_rate: float = 0.1, mutation_power: float = 0.1) -> Genome:
        """Create a mutated copy of this genome."""
        new_weights = []
        new_biases = []

        for w in self.weights:
            mutation = np.random.randn(*w.shape) * mutation_power
            mask = np.random.random(w.shape) < mutation_rate
            new_w = w + (mutation * mask)
            new_weights.append(new_w.astype(np.float32))

        for b in self.biases:
            mutation = np.random.randn(*b.shape) * mutation_power
            mask = np.random.random(b.shape) < mutation_rate
            new_b = b + (mutation * mask)
            new_biases.append(new_b.astype(np.float32))

        return Genome(new_weights, new_biases, generation=self.generation + 1)

    def crossover(self, other: Genome) -> Genome:
        """Uniform crossover with another genome."""
        child_weights = []
        child_biases = []

        for w1, w2 in zip(self.weights, other.weights):
            mask = np.random.random(w1.shape) < 0.5
            child_w = np.where(mask, w1, w2)
            child_weights.append(child_w.astype(np.float32))

        for b1, b2 in zip(self.biases, other.biases):
            mask = np.random.random(b1.shape) < 0.5
            child_b = np.where(mask, b1, b2)
            child_biases.append(child_b.astype(np.float32))

        return Genome(
            child_weights, child_biases,
            generation=max(self.generation, other.generation) + 1
        )

    def save(self, filepath: str) -> None:
        """Save genome to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'biases': self.biases,
                'fitness': self.fitness,
                'generation': self.generation
            }, f)

    @classmethod
    def load(cls, filepath: str) -> 'Genome':
        """Load genome from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        genome = cls(data['weights'], data['biases'])
        genome.fitness = data['fitness']
        genome.generation = data['generation']
        return genome

    @classmethod
    def random(cls, layer_sizes: List[int]) -> 'Genome':
        """Create a random genome with Xavier initialization."""
        weights = []
        biases = []

        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            limit = np.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i + 1]))
            w = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1]))
            b = np.zeros(layer_sizes[i + 1])
            weights.append(w.astype(np.float32))
            biases.append(b.astype(np.float32))

        return cls(weights, biases)


class NeuralNetwork:
    """Simple feedforward neural network."""

    def __init__(self, genome: Genome):
        self.genome = genome
        self.weights = genome.weights
        self.biases = genome.biases

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = np.tanh(x @ w + b)
        # Last layer uses softmax
        x = x @ self.weights[-1] + self.biases[-1]
        x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        x = x / (np.sum(x, axis=-1, keepdims=True) + 1e-8)
        return x

    def act(self, observation: np.ndarray, legal_actions: np.ndarray) -> int:
        """
        Choose action based on observation and legal actions.

        Args:
            observation: [features] or [batch, features]
            legal_actions: [3] boolean array of legal actions

        Returns:
            Action index (0-2)
        """
        if observation.ndim == 1:
            observation = observation[None, :]

        # Get action probabilities
        probs = self.forward(observation)[0]

        # Mask illegal actions
        if np.any(legal_actions):
            probs = probs * legal_actions.astype(np.float32)
            probs = probs / (np.sum(probs) + 1e-8)

        # Choose action
        return np.random.choice(3, p=probs)


class GreedyController:
    """
    Simple greedy AI that tries to maximize survival and trap opponents.
    """

    @staticmethod
    def act(observation: np.ndarray, legal_actions: np.ndarray,
            pos: np.ndarray, heading: int, occupied: np.ndarray) -> int:
        """
        Choose action using greedy heuristic.

        Strategy:
        1. Prefer actions that don't lead to immediate death
        2. Try to move towards the center
        3. If all else fails, pick first legal action
        """
        # Get legal actions that are safe for next step
        safe_actions = np.where(legal_actions)[0]

        if len(safe_actions) == 0:
            return 0  # No safe actions, pick straight

        # Try to move towards center
        center = np.array([occupied.shape[1] / 2, occupied.shape[0] / 2])
        current_pos = pos[-2:] if len(pos.shape) > 1 else pos

        best_action = safe_actions[0]
        best_distance = float('inf')

        for action in safe_actions:
            # Calculate next position based on action
            delta = GreedyController._get_delta(heading, action)
            next_pos = current_pos + delta

            distance = np.linalg.norm(next_pos - center)
            if distance < best_distance:
                best_distance = distance
                best_action = action

        return best_action

    @staticmethod
    def _get_delta(heading: int, action: int) -> np.ndarray:
        """Calculate position delta for given heading and action."""
        # Heading: 0 up, 1 right, 2 down, 3 left
        # Action: 0 straight, 1 left, 2 right
        turn_map = {
            (0, 0): (0, -1), (0, 1): (-1, 0), (0, 2): (1, 0),
            (1, 0): (1, 0), (1, 1): (0, -1), (1, 2): (0, 1),
            (2, 0): (0, 1), (2, 1): (1, 0), (2, 2): (-1, 0),
            (3, 0): (-1, 0), (3, 1): (0, 1), (3, 2): (0, -1)
        }
        return np.array(turn_map[(heading, action)])


class TrainingConfig:
    """Configuration for training."""

    def __init__(self, config_dict: Optional[Dict] = None):
        # Environment config
        self.width = 32
        self.height = 32
        self.max_steps = 500
        self.players = 2

        # Population config
        self.population_size = 100
        self.elite_ratio = 0.2
        self.crossover_ratio = 0.6
        self.mutation_ratio = 0.2
        self.mutation_rate = 0.1
        self.mutation_power = 0.1

        # Training config
        self.generations = 1000
        self.episodes_per_genome = 10  # Number of episodes to evaluate each genome
        self.batch_envs = 50  # Parallel environments for evaluation

        # Opponent config
        self.train_against_self = True
        self.train_against_greedy = True
        self.greedy_ratio = 0.3  # Ratio of opponents that are greedy

        # Neural network config
        self.input_features = self._calculate_input_features()
        self.hidden_layers = [128, 64]
        self.output_size = 3

        # Checkpoint config
        self.checkpoint_dir = "checkpoints"
        self.save_best_every = 10
        self.save_population_every = 50

        if config_dict:
            self.__dict__.update(config_dict)

    def _calculate_input_features(self) -> int:
        """Calculate number of input features for the neural network."""
        # This should match the output of TronBatchModel.observe_lite()
        # Base features per player: legal(3) + pos(2) + heading_oh(4) + alive(1) = 10
        # Plus relative info for other players: (pos(2) + alive(1)) * (players - 1)
        players = 4  # Maximum players
        base_features = 10
        rel_features = 3 * (players - 1)
        return base_features + rel_features

    @classmethod
    def from_file(cls, filepath: str) -> 'TrainingConfig':
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)

    def save(self, filepath: str) -> None:
        """Save config to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=2)


class TronTrainer:
    """
    Genetic Algorithm trainer for Tron agents.
    """

    def __init__(
            self,
            config: TrainingConfig,
            population: Optional[List[Genome]] = None,
            generation: int = 0
    ):
        self.config = config
        self.population = population or []
        self.generation = generation
        self.best_genome = None
        self.best_fitness = -float('inf')
        self.history = []

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Initialize population if needed
        if not self.population:
            self._initialize_population()

    def _initialize_population(self) -> None:
        """Initialize random population."""
        layer_sizes = [
            self.config.input_features,
            *self.config.hidden_layers,
            self.config.output_size
        ]

        print(f"Initializing population of {self.config.population_size} genomes...")
        print(f"Network architecture: {layer_sizes}")

        for _ in range(self.config.population_size):
            genome = Genome.random(layer_sizes)
            self.population.append(genome)

    def evaluate_genome(
            self,
            genome: Genome,
            model: TronBatchModel,
            opponent_genomes: List[Genome],
            use_greedy: bool = True
    ) -> float:
        """
        Evaluate a single genome against opponents.

        Args:
            genome: Genome to evaluate
            model: Tron environment model
            opponent_genomes: List of opponent genomes to play against
            use_greedy: Whether to include greedy opponents

        Returns:
            Average fitness score
        """
        total_score = 0.0
        network = NeuralNetwork(genome)

        for episode in range(self.config.episodes_per_genome):
            # Reset environments
            model.reset()

            # Setup opponents
            opponents = []
            if use_greedy and self.config.train_against_greedy:
                # Add greedy opponents
                num_greedy = int(self.config.greedy_ratio * (model.players - 1))
                opponents.extend([None] * num_greedy)  # None indicates greedy

            # Fill remaining with opponent genomes
            remaining = (model.players - 1) - len(opponents)
            for i in range(remaining):
                opp_idx = np.random.randint(len(opponent_genomes))
                opponents.append(opponent_genomes[opp_idx])

            # Shuffle opponents
            np.random.shuffle(opponents)

            # Run episode
            episode_score = self._run_episode(
                model, network, opponents, episode_id=episode
            )
            total_score += episode_score

        return total_score / self.config.episodes_per_genome

    def _run_episode(
            self,
            model: TronBatchModel,
            agent_network: NeuralNetwork,
            opponents: List[Optional[Genome]],
            episode_id: int = 0
    ) -> float:
        """
        Run a single episode with the agent against opponents.

        Args:
            model: Tron environment model
            agent_network: Neural network for the main agent
            opponents: List of opponent genomes (None for greedy)

        Returns:
            Score for the agent (1 for win, 0 for loss, 0.5 for draw)
        """
        # Always set agent as player 0
        player_id = 0

        # Initialize opponent networks
        opponent_nets = []
        for opp_genome in opponents:
            if opp_genome is None:
                opponent_nets.append(None)  # Greedy
            else:
                opponent_nets.append(NeuralNetwork(opp_genome))

        # Run simulation
        while not np.any(model.done):
            # Get observations and legal actions for all players
            obs = model.observe_lite()  # [1, players, features]
            legal = model.legal_actions()  # [1, players, 3]

            # Choose actions
            actions = np.zeros((1, model.players), dtype=np.int8)

            # Agent action
            agent_obs = obs[0, player_id]
            agent_legal = legal[0, player_id]
            actions[0, player_id] = agent_network.act(agent_obs, agent_legal)

            # Opponent actions
            for opp_id, (opp_net, opp_genome) in enumerate(zip(opponent_nets, opponents)):
                actual_id = opp_id + 1 if opp_id < player_id else opp_id + 1
                opp_obs = obs[0, actual_id]
                opp_legal = legal[0, actual_id]

                if opp_net is None:  # Greedy
                    # Greedy needs position and heading info
                    pos = model.pos[0, actual_id]
                    heading = model.heading[0, actual_id]
                    occupied = model.occupied[0]
                    actions[0, actual_id] = GreedyController.act(
                        opp_obs, opp_legal, pos, heading, occupied
                    )
                else:  # Neural network opponent
                    actions[0, actual_id] = opp_net.act(opp_obs, opp_legal)

            # Take step
            result = model.step(actions)

            # Check if episode ended
            if np.any(result.done):
                break

        # Calculate score
        if model.alive[0, player_id]:
            # Agent survived, check if others died
            others_alive = model.alive[0, [i for i in range(model.players) if i != player_id]]
            if not np.any(others_alive):
                return 1.0  # Win
            else:
                return 0.5  # Draw (agent alive but others also alive - max steps reached)
        else:
            return 0.0  # Loss

    def evaluate_population(self, verbose: bool = True) -> None:
        """
        Evaluate all genomes in the population.
        """
        if verbose:
            print(f"\nEvaluating generation {self.generation}...")

        # Create environment for evaluation
        model = TronBatchModel(
            width=self.config.width,
            height=self.config.height,
            players=4,  # Fixed 4 players for evaluation
            envs=1,
            max_steps=self.config.max_steps,
            keep_owner=False
        )

        # Evaluate each genome
        for idx, genome in enumerate(self.population):
            # Select opponents from population (excluding current)
            opponent_indices = list(range(len(self.population)))
            opponent_indices.remove(idx)
            opponent_genomes = [self.population[i] for i in
                                np.random.choice(opponent_indices, size=min(10, len(opponent_indices)))]

            fitness = self.evaluate_genome(genome, model, opponent_genomes)
            genome.fitness = fitness

            if verbose and idx % 20 == 0:
                print(f"  Evaluated {idx + 1}/{len(self.population)} genomes")

        # Sort by fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)

        # Update best genome
        if self.population[0].fitness > self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.best_genome = self.population[0]
            if verbose:
                print(f"  New best fitness: {self.best_fitness:.3f}")

        # Record history
        self.history.append({
            'generation': self.generation,
            'best_fitness': self.population[0].fitness,
            'avg_fitness': np.mean([g.fitness for g in self.population]),
            'std_fitness': np.std([g.fitness for g in self.population])
        })

        if verbose:
            avg_fitness = np.mean([g.fitness for g in self.population])
            print(f"  Generation {self.generation} - Best: {self.population[0].fitness:.3f}, "
                  f"Avg: {avg_fitness:.3f}")

    def evolve_population(self) -> None:
        """
        Create new population through selection, crossover, and mutation.
        """
        new_population = []

        # Elitism: keep top performers
        elite_size = int(self.config.population_size * self.config.elite_ratio)
        new_population.extend(self.population[:elite_size])

        # Crossover: create offspring from pairs of parents
        crossover_size = int(self.config.population_size * self.config.crossover_ratio)
        for _ in range(crossover_size):
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()
            child = parent1.crossover(parent2)
            new_population.append(child)

        # Mutation: create mutated copies of remaining
        mutation_size = self.config.population_size - len(new_population)
        for _ in range(mutation_size):
            parent = self._tournament_select()
            child = parent.mutate(
                self.config.mutation_rate,
                self.config.mutation_power
            )
            new_population.append(child)

        self.population = new_population
        self.generation += 1

    def _tournament_select(self, tournament_size: int = 3) -> Genome:
        """Select a genome using tournament selection."""
        indices = np.random.choice(len(self.population), tournament_size, replace=False)
        best_idx = max(indices, key=lambda i: self.population[i].fitness)
        return self.population[best_idx]

    def train(self, generations: Optional[int] = None) -> None:
        """
        Run the training loop.

        Args:
            generations: Number of generations to train (uses config if None)
        """
        generations = generations or self.config.generations
        start_time = time.time()

        print("=" * 80)
        print("Starting Tron Genetic Algorithm Training")
        print("=" * 80)
        print(f"Population size: {self.config.population_size}")
        print(f"Generations: {generations}")
        print(f"Train against self: {self.config.train_against_self}")
        print(f"Train against greedy: {self.config.train_against_greedy}")
        print("=" * 80)

        for gen in range(generations):
            # Evaluate current population
            self.evaluate_population(verbose=True)

            # Save checkpoint if needed
            if (self.generation % self.config.save_best_every == 0 and
                    self.best_genome is not None):
                self.save_checkpoint()

            # Evolve for next generation (except last)
            if gen < generations - 1:
                self.evolve_population()

            # Print progress
            elapsed = time.time() - start_time
            avg_fitness = self.history[-1]['avg_fitness']
            print(f"  Time elapsed: {elapsed:.1f}s, Avg fitness: {avg_fitness:.3f}")

        # Final save
        self.save_checkpoint(final=True)
        print("\nTraining completed!")
        print(f"Best fitness achieved: {self.best_fitness:.3f}")
        print(f"Total time: {time.time() - start_time:.1f}s")

    def save_checkpoint(self, final: bool = False) -> None:
        """Save current training state."""
        checkpoint = {
            'generation': self.generation,
            'population': self.population,
            'best_genome': self.best_genome,
            'best_fitness': self.best_fitness,
            'history': self.history,
            'config': self.config.__dict__
        }

        if final:
            filename = f"final_checkpoint_gen_{self.generation}.pkl"
        else:
            filename = f"checkpoint_gen_{self.generation}.pkl"

        filepath = Path(self.config.checkpoint_dir) / filename
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"  Checkpoint saved: {filepath}")

        # Also save best genome separately
        if self.best_genome:
            best_path = Path(self.config.checkpoint_dir) / "best_genome.pkl"
            self.best_genome.save(str(best_path))

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str) -> 'TronTrainer':
        """Load training state from checkpoint."""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)

        config = TrainingConfig(checkpoint['config'])
        trainer = cls(
            config=config,
            population=checkpoint['population'],
            generation=checkpoint['generation']
        )
        trainer.best_genome = checkpoint['best_genome']
        trainer.best_fitness = checkpoint['best_fitness']
        trainer.history = checkpoint['history']

        print(f"Loaded checkpoint from generation {trainer.generation}")
        print(f"Best fitness: {trainer.best_fitness:.3f}")

        return trainer

    def export_best_genome(self, filepath: str) -> None:
        """Export the best genome to a file."""
        if self.best_genome:
            self.best_genome.save(filepath)
            print(f"Best genome exported to {filepath}")
        else:
            print("No best genome available")

    def get_best_network(self) -> Optional[NeuralNetwork]:
        """Get the neural network of the best genome."""
        if self.best_genome:
            return NeuralNetwork(self.best_genome)
        return None


def create_training_script():
    """Create an example training script."""
    script_content = '''
#!/usr/bin/env python3
"""
Example training script for Tron agents.
"""

from trainer import TronTrainer, TrainingConfig

def main():
    # Configure training
    config = TrainingConfig()
    config.population_size = 50
    config.generations = 100
    config.episodes_per_genome = 5
    config.train_against_self = True
    config.train_against_greedy = True
    config.greedy_ratio = 0.3
    config.save_best_every = 10

    # Create trainer
    trainer = TronTrainer(config)

    # Train
    trainer.train()

    # Export best genome
    trainer.export_best_genome("best_agent.pkl")

    print("\\nTraining complete! Best agent saved to best_agent.pkl")

if __name__ == "__main__":
    main()
'''

    with open("example_train.py", "w") as f:
        f.write(script_content)

    print("Created example training script: example_train.py")


if __name__ == "__main__":
    # Example usage
    create_training_script()

    # Quick test with small population
    print("\nRunning quick test...")
    config = TrainingConfig()
    config.population_size = 10
    config.generations = 3
    config.episodes_per_genome = 2

    trainer = TronTrainer(config)
    trainer.train()

    print("\nTest completed successfully!")