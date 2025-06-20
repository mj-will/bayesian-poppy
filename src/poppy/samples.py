from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from array_api_compat import (
    array_namespace,
    is_jax_array,
    is_numpy_namespace,
    to_device,
)
from array_api_compat import device as api_device
from array_api_compat.common._typing import Array

from .utils import logsumexp, recursively_save_to_h5_file, to_numpy

logger = logging.getLogger(__name__)


@dataclass
class BaseSamples:
    """Class for storing samples and corresponding weights.

    If :code:`xp` is not specified, all inputs will be converted to match
    the array type of :code:`x`.
    """

    x: Array
    log_likelihood: Array | None = None
    log_prior: Array | None = None
    log_q: Array | None = None
    parameters: list[str] | None = None
    xp: Callable | None = None
    device: Any = None

    def __post_init__(self):
        if self.xp is None:
            self.xp = array_namespace(self.x)
        # Numpy arrays need to be on the CPU before being converted
        if is_numpy_namespace(self.xp):
            self.device = "cpu"
        self.x = self.array_to_namespace(self.x)
        if self.device is None:
            self.device = api_device(self.x)
        if self.log_likelihood is not None:
            self.log_likelihood = self.array_to_namespace(self.log_likelihood)
        if self.log_prior is not None:
            self.log_prior = self.array_to_namespace(self.log_prior)
        if self.log_q is not None:
            self.log_q = self.array_to_namespace(self.log_q)

        if self.parameters is None:
            self.parameters = [f"x_{i}" for i in range(self.dims)]

    @property
    def dims(self):
        """Number of dimensions (parameters) in the samples."""
        if self.x is None:
            return 0
        return self.x.shape[1] if self.x.ndim > 1 else 1

    def to_numpy(self):
        return self.__class__(
            x=to_numpy(self.x),
            parameters=self.parameters,
            log_likelihood=to_numpy(self.log_likelihood)
            if self.log_likelihood is not None
            else None,
            log_prior=to_numpy(self.log_prior)
            if self.log_prior is not None
            else None,
            log_q=to_numpy(self.log_q) if self.log_q is not None else None,
        )

    def to_namespace(self, xp):
        return self.__class__(
            x=xp.asarray(self.x),
            parameters=self.parameters,
            log_likelihood=xp.asarray(self.log_likelihood)
            if self.log_likelihood is not None
            else None,
            log_prior=xp.asarray(self.log_prior)
            if self.log_prior is not None
            else None,
            log_q=xp.asarray(self.log_q) if self.log_q is not None else None,
        )

    def array_to_namespace(self, x):
        """Convert an array to the same namespace as the samples"""
        if is_numpy_namespace(self.xp) and not is_jax_array(x):
            x = to_device(x, "cpu")
        x = self.xp.asarray(x)
        if self.device:
            x = to_device(x, self.device)
        return x

    def to_dict(self, flat: bool = True):
        samples = dict(zip(self.parameters, self.x.T, strict=True))
        out = {
            "log_likelihood": self.log_likelihood,
            "log_prior": self.log_prior,
            "log_q": self.log_q,
        }
        if flat:
            out.update(samples)
        else:
            out["samples"] = samples
        return out

    def to_dataframe(self, flat: bool = True):
        import pandas as pd

        return pd.DataFrame(self.to_dict(flat=flat))

    def plot_corner(self, **kwargs):
        import corner

        kwargs = copy.deepcopy(kwargs)
        kwargs.setdefault("labels", self.parameters)
        fig = corner.corner(to_numpy(self.x), **kwargs)
        return fig

    def __str__(self):
        out = (
            f"No. samples: {len(self.x)}\n"
            f"No. parameters: {len(self.parameters)}\n"
        )
        return out

    def save(self, h5_file, path="samples", flat=False):
        """Save the samples to an HDF5 file.

        This converts the samples to numpy and then to a dictionary.

        Parameters
        ----------
        h5_file : h5py.File
            The HDF5 file to save to.
        path : str
            The path in the HDF5 file to save to.
        flat : bool
            If True, save the samples as a flat dictionary.
            If False, save the samples as a nested dictionary.
        """
        dictionary = self.to_numpy().to_dict(flat=flat)
        recursively_save_to_h5_file(h5_file, path, dictionary)

    def __len__(self):
        return len(self.x)


@dataclass
class Samples(BaseSamples):
    """Class for storing samples and corresponding weights.

    If :code:`xp` is not specified, all inputs will be converted to match
    the array type of :code:`x`.
    """

    log_w: Array = field(init=False)
    weights: Array = field(init=False)
    evidence: float = field(init=False)
    evidence_error: float = field(init=False)
    log_evidence: float | None = None
    log_evidence_error: float | None = None
    effective_sample_size: float = field(init=False)

    def __post_init__(self):
        super().__post_init__()

        if all(
            x is not None
            for x in [self.log_likelihood, self.log_prior, self.log_q]
        ):
            self.compute_weights()
        else:
            self.log_w = None
            self.weights = None
            self.evidence = None
            self.evidence_error = None
            self.effective_sample_size = None

    @property
    def efficiency(self):
        """Efficiency of the weighted samples.

        Defined as ESS / number of samples.
        """
        if self.log_w is None:
            raise RuntimeError("Samples do not contain weights!")
        return self.effective_sample_size / len(self.x)

    def compute_weights(self):
        """Compute the posterior weights."""
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

    @property
    def scaled_weights(self):
        return self.xp.exp(self.log_w - self.xp.max(self.log_w))

    def rejection_sample(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        log_u = self.xp.asarray(
            np.log(rng.uniform(size=len(self.x))), device=self.device
        )
        log_w = self.log_w - self.xp.max(self.log_w)
        accept = log_w > log_u
        return self.__class__(
            x=self.x[accept],
            log_likelihood=self.log_likelihood[accept],
            log_prior=self.log_prior[accept],
        )

    def to_dict(self, flat: bool = True):
        samples = dict(zip(self.parameters, self.x.T, strict=True))
        out = super().to_dict(flat=flat)
        other = {
            "log_w": self.log_w,
            "weights": self.weights,
            "evidence": self.evidence,
            "log_evidence": self.log_evidence,
            "evidence_error": self.evidence_error,
            "log_evidence_error": self.log_evidence_error,
            "effective_sample_size": self.effective_sample_size,
        }
        out.update(other)
        if flat:
            out.update(samples)
        else:
            out["samples"] = samples
        return out

    def plot_corner(self, include_weights: bool = True, **kwargs):
        kwargs = copy.deepcopy(kwargs)
        if (
            include_weights
            and self.weights is not None
            and "weights" not in kwargs
        ):
            kwargs["weights"] = to_numpy(self.scaled_weights)
        return super().plot_corner(**kwargs)

    def __str__(self):
        out = super().__str__()
        if self.log_evidence is not None:
            out += f"Log evidence: {self.log_evidence:.2f} +/- {self.log_evidence_error:.2f}\n"
        if self.log_w is not None:
            out += (
                f"Effective sample size: {self.effective_sample_size:.1f}\n"
                f"Efficiency: {self.efficiency:.2f}\n"
            )
        return out

    def to_namespace(self, xp):
        return self.__class__(
            x=xp.asarray(self.x),
            parameters=self.parameters,
            log_likelihood=xp.asarray(self.log_likelihood)
            if self.log_likelihood is not None
            else None,
            log_prior=xp.asarray(self.log_prior)
            if self.log_prior is not None
            else None,
            log_q=xp.asarray(self.log_q) if self.log_q is not None else None,
            log_evidence=xp.asarray(self.log_evidence)
            if self.log_evidence is not None
            else None,
            log_evidence_error=xp.asarray(self.log_evidence_error)
            if self.log_evidence_error is not None
            else None,
        )

    def to_numpy(self):
        return self.__class__(
            x=to_numpy(self.x),
            parameters=self.parameters,
            log_likelihood=to_numpy(self.log_likelihood)
            if self.log_likelihood is not None
            else None,
            log_prior=to_numpy(self.log_prior)
            if self.log_prior is not None
            else None,
            log_q=to_numpy(self.log_q) if self.log_q is not None else None,
            log_evidence=self.log_evidence
            if self.log_evidence is not None
            else None,
            log_evidence_error=self.log_evidence_error
            if self.log_evidence_error is not None
            else None,
        )


@dataclass
class SMCSamples(BaseSamples):
    beta: float | None = None
    log_evidence: float | None = None
    """Temperature parameter for the current samples."""

    def log_p_t(self, beta):
        log_p_T = self.log_likelihood + self.log_prior
        return (1 - beta) * self.log_q + beta * log_p_T

    def unnormalized_log_weights(self, beta):
        return (self.beta - beta) * self.log_q + (beta - self.beta) * (
            self.log_likelihood + self.log_prior
        )

    def log_evidence_ratio(self, beta):
        log_w = self.unnormalized_log_weights(beta)
        return logsumexp(log_w) - math.log(len(self.x))

    def log_weights(self, beta) -> Array:
        log_w = self.unnormalized_log_weights(beta)
        if self.xp.isnan(log_w).any():
            raise ValueError(f"Log weights contain NaN values for beta={beta}")
        log_evidence_ratio = logsumexp(log_w) - math.log(len(self.x))
        return log_w + log_evidence_ratio

    def resample(self, beta, n_samples: int | None = None) -> "SMCSamples":
        if beta == self.beta:
            logger.warning("Resampling with the same beta value")
            return self
        if n_samples is None:
            n_samples = len(self.x)
        log_w = self.log_weights(beta)
        w = to_numpy(self.xp.exp(log_w - logsumexp(log_w)))
        idx = np.random.choice(len(self.x), size=n_samples, replace=True, p=w)
        return self.__class__(
            x=self.x[idx],
            log_likelihood=self.log_likelihood[idx],
            log_prior=self.log_prior[idx],
            log_q=self.log_q[idx],
            beta=beta,
        )

    def __str__(self):
        out = super().__str__()
        if self.log_evidence is not None:
            out += f"Log evidence: {self.log_evidence:.2f}\n"
        return out

    def to_standard_samples(self):
        """Convert the samples to standard samples."""
        return Samples(
            x=self.x,
            log_likelihood=self.log_likelihood,
            log_prior=self.log_prior,
            xp=self.xp,
            parameters=self.parameters,
            log_evidence=self.log_evidence,
            log_evidence_error=self.log_evidence_error,
        )
