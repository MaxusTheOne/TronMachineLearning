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