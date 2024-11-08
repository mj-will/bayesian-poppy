import copy
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from array_api_compat import array_namespace
from array_api_compat.common._typing import Array
from scipy.special import logsumexp


@dataclass
class Samples:
    x: Array
    parameters: list[str] | None = None
    log_likelihood: Array | None = None
    log_prior: Array | None = None
    log_q: Array | None = None
    log_w: Array = field(init=False)
    weights: Array = field(init=False)
    evidence: float = field(init=False)
    evidence_error: float = field(init=False)
    log_evidence: float = field(init=False)
    log_evidence_error: float = field(init=False)
    effective_sample_size: float = field(init=False)
    xp: Callable | None = None

    def __post_init__(self):
        if self.xp is None:
            self.xp = array_namespace(self.x)
        else:
            if self.log_likelihood is not None:
                self.log_likelihood = self.xp.asarray(self.log_likelihood)
            if self.log_prior is not None:
                self.log_prior = self.xp.asarray(self.log_prior)
            if self.log_q is not None:
                self.log_q = self.xp.asarray(self.log_q)

        if self.parameters is None:
            self.parameters = [f"x_{i}" for i in range(len(self.x[0]))]
        if all(
            x is not None
            for x in [self.log_likelihood, self.log_prior, self.log_q]
        ):
            self.compute_weights()
        else:
            self.log_w = None
            self.weights = None
            self.evidence = None
            self.log_evidence = None
            self.evidence_error = None
            self.log_evidence_error = None

    @property
    def efficiency(self):
        return self.effective_sample_size / len(self.x)

    def compute_weights(self):
        self.log_w = self.log_likelihood + self.log_prior - self.log_q
        self.log_evidence = self.xp.asarray(logsumexp(self.log_w)) - math.log(
            len(self.x)
        )
        self.weights = self.xp.exp(self.log_w)
        self.evidence = self.xp.exp(self.log_evidence)
        n = len(self.x)
        self.evidence_error = self.xp.sqrt(
            self.xp.sum((self.weights - self.evidence) ** 2) / (n * (n - 1))
        )
        self.log_evidence_error = self.xp.abs(
            self.evidence_error / self.evidence
        )
        log_w = self.log_w - self.xp.max(self.log_w)
        self.effective_sample_size = self.xp.exp(
            self.xp.asarray(logsumexp(log_w) * 2 - logsumexp(log_w * 2))
        )
        self.effective_number = self.xp.exp(self.xp.asarray(logsumexp(log_w)))

    @property
    def scaled_weights(self):
        return self.xp.exp(self.log_w - self.xp.max(self.log_w))

    def rejection_sample(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        log_u = self.xp.asarray(np.log(rng.uniform(size=len(self.x))))
        log_w = self.log_w - self.xp.nanmax(self.log_w)
        accept = log_w > log_u
        return self.__class__(
            x=self.x[accept],
            log_likelihood=self.log_likelihood[accept],
            log_prior=self.log_prior[accept],
        )

    def to_numpy(self):
        return self.__class__(
            x=np.asarray(self.x),
            parameters=self.parameters,
            log_likelihood=np.asarray(self.log_likelihood),
            log_prior=np.asarray(self.log_prior),
            log_q=np.asarray(self.log_q),
        )

    def to_dict(self, flat: bool = True):
        samples = dict(zip(self.parameters, self.x.T, strict=True))
        out = {
            "log_likelihood": self.log_likelihood,
            "log_prior": self.log_prior,
            "log_q": self.log_q,
            "log_w": self.log_w,
            "weights": self.weights,
            "evidence": self.evidence,
            "log_evidence": self.log_evidence,
            "evidence_error": self.evidence_error,
            "log_evidence_error": self.log_evidence_error,
            "effective_sample_size": self.effective_sample_size,
        }
        if flat:
            out.update(samples)
        else:
            out["samples"] = samples
        return out

    def to_dataframe(self, flat: bool = True):
        import pandas as pd

        return pd.DataFrame(self.to_dict(flat=flat))

    def plot_corner(self, include_weights: bool = True, **kwargs):
        kwargs = copy.deepcopy(kwargs)
        import corner

        if include_weights and self.weights is not None:
            kwargs["weights"] = np.asarray(self.scaled_weights)
        kwargs.setdefault("labels", self.parameters)
        fig = corner.corner(np.asarray(self.x), **kwargs)
        return fig

    def __str__(self):
        return (
            f"No. samples: {len(self.x)}\n"
            f"No. parameters: {len(self.parameters)}\n"
            f"Log evidence: {self.log_evidence:.2f} +/- {self.log_evidence_error:.2f}\n"
            f"Effective sample size: {self.effective_sample_size:.1f}\n"
            f"Efficiency: {self.efficiency:.2f}\n"
            f"Effective no. of samples: {self.effective_number:.2f}"
        )


def torch_to_numpy(value, /, device=None, dtype=None):
    return value.detach().numpy() if value is not None else None


def jax_to_numpy(value, /, device=None, dtype=None):
    return np.array(value, dtype=None) if value is not None else None
