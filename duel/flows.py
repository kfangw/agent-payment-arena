"""Payment flows: the joint distribution of exposure and suspicion.

A flow is generated per payment as: true risk p ~ D, intent
theta ~ Bernoulli(p), score pi0 = p (calibrated) or g(p) (overconfident
distortion, pushing mass to both ends on the logit scale).  The base
rate b is E[p], the unconditional probability that a payment's hidden
intent is misuse.

DRAFT STATUS.  The numbers in FLOW_SPEC are a first draft: three base
rates, two D families, one distortion.  They are not settled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    return np.log(p) - np.log1p(-p)


# --------------------------------------------------------------- D families
@dataclass
class MixD:
    """Bimodal mixture: a benign cluster plus a small adversarial cluster.
    The two Beta components are fixed; the base rate b sets the weight."""

    b: float
    a0: float = 1.0
    b0: float = 499.0  # benign cluster, mean 0.002
    a1: float = 17.0
    b1: float = 3.0  # adversarial cluster, mean 0.85

    def __post_init__(self):
        mu0 = self.a0 / (self.a0 + self.b0)
        mu1 = self.a1 / (self.a1 + self.b1)
        self.w = (self.b - mu0) / (mu1 - mu0)
        assert 0.0 <= self.w <= 1.0, "base rate outside the mixture's span"

    def sample(self, n, rng):
        hot = rng.random(n) < self.w
        p = np.where(hot, rng.beta(self.a1, self.b1, size=n), rng.beta(self.a0, self.b0, size=n))
        return np.clip(p, 1e-9, 1 - 1e-9)


@dataclass
class UniD:
    """Unimodal Beta with mean b and concentration kappa: thick mass
    around the decision boundary when b sits near pi-hat."""

    b: float
    kappa: float = 8.0

    def sample(self, n, rng):
        p = rng.beta(self.kappa * self.b, self.kappa * (1.0 - self.b), size=n)
        return np.clip(p, 1e-9, 1 - 1e-9)


# --------------------------------------------------------------- distortion
def overconfident(p, gamma):
    """g(p): push scores to both ends on the logit scale."""
    return np.clip(_expit(gamma * _logit(p)), 1e-9, 1 - 1e-9)


# --------------------------------------------------------------- exposure
@dataclass
class LogNormalV:
    """Exposure in dollars: log-normal, declared range by truncation."""

    median: float = 30.0
    sigma: float = 1.0
    lo: float = 0.5
    hi: float = 2000.0

    def sample(self, n, rng):
        v = np.exp(np.log(self.median) + self.sigma * rng.standard_normal(n))
        return np.clip(v, self.lo, self.hi)


# --------------------------------------------------------------- flows
@dataclass
class Flow:
    name: str
    D: object
    vdist: LogNormalV
    gamma: float | None = None  # None = calibrated scorer

    def sample(self, n, rng):
        v = self.vdist.sample(n, rng)
        p = self.D.sample(n, rng)
        theta = (rng.random(n) < p).astype(np.int8)
        pi0 = p.copy() if self.gamma is None else overconfident(p, self.gamma)
        return v, p, theta, pi0


# Draft numbers; see the spec note for the rationale.
FLOW_SPEC = dict(
    b_low=0.005,
    b_mid=0.05,
    b_high=0.20,
    gamma=2.0,
    v_median=30.0,
    v_sigma=1.0,
    v_range=(0.5, 2000.0),
)


def make_flows(spec=FLOW_SPEC):
    vd = LogNormalV(spec["v_median"], spec["v_sigma"], *spec["v_range"])
    return {
        "F1": Flow("F1", MixD(spec["b_low"]), vd),
        "F2": Flow("F2", UniD(spec["b_high"]), vd),
        "F3": Flow("F3", MixD(spec["b_mid"]), vd, gamma=spec["gamma"]),
    }
