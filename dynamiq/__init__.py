"""Dynamiq -- RL algorithms as a typed, compiled graph.

You don't write a training loop. You declare an algorithm as a graph of typed
signals; ``dq.compile`` type-checks it (catching missing stop-grads, off-policy
data in on-policy losses, unsynced target nets) and returns a runnable loop.

The same vocabulary expresses different algorithm families:
  * DQN     -- value-based, off-policy        (examples/dqn_cartpole.py)
  * PPO     -- policy-gradient, on-policy      (examples/ppo_cartpole.py)
  * SAC     -- actor-critic, off-policy        (examples/sac_discrete_cartpole.py)
"""

from . import dist
from . import losses as loss
from .compiler import Algorithm, compile
from .dist import Categorical
from .exceptions import DynamiqConfigError, DynamiqError, DynamiqTypeError
from .networks import Network, PolicyNetwork, QNetwork
from .optim import SGD, Adam, OptimizerSpec
from .params import Parameter
from .signal import Provenance, Signal, maximum, minimum
from .sources import ReplayBuffer, ReplaySample, RolloutBuffer, RolloutSample
from .targets import Target
from .verify import Finding, explain, verify

__all__ = [
    "loss",
    "dist",
    "compile",
    "Algorithm",
    "QNetwork",
    "PolicyNetwork",
    "Network",
    "Parameter",
    "Categorical",
    "Target",
    "ReplayBuffer",
    "ReplaySample",
    "RolloutBuffer",
    "RolloutSample",
    "Adam",
    "SGD",
    "OptimizerSpec",
    "Signal",
    "Provenance",
    "minimum",
    "maximum",
    "verify",
    "explain",
    "Finding",
    "DynamiqError",
    "DynamiqTypeError",
    "DynamiqConfigError",
]

__version__ = "0.1.0"
