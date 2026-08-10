"""An evaluation environment for LLM agents that hold payment authority.

The arena runs an agent against a scenario, lets it reach a paid resource
through a gateway, and scores what it spent against what it was authorized to
spend. Everything below the CLI is deliberately split into two halves: the
*subject* (agent, gateway, policy) and the *instrument* (scenario, metrics,
report). Only the instrument knows the ground truth of a run.
"""

__version__ = "0.0.1"

__all__ = ["__version__"]
