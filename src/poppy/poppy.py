import logging
from typing import Callable

from .flows import get_flow_wrapper
from .samples import Samples
from .transforms import DataTransform

logger = logging.getLogger(__name__)


class Poppy:
    """Posterior post-processing.

    Parameters
    ----------
    log_likelihood : Callable
        The log likelihood function.
    log_prior : Callable
        The log prior function.
    dims : int
        The number of dimensions.
    flow_matching : bool
        Whether to use flow matching.
    **kwargs
        Keyword arguments to pass to the flow.
    """

    def __init__(
        self,
        *,
        log_likelihood: Callable,
        log_prior: Callable,
        dims: int,
        parameters: list[str] | None = None,
        periodic_parameters: list[str] | None = None,
        prior_bounds: dict[str, tuple[float, float]] | None = None,
        bounded_to_unbounded: bool = True,
        flow_matching: bool = False,
        device: str | None = None,
        xp: None = None,
        flow_backend: str = "zuko",
        **kwargs,
    ) -> None:
        self.log_likelihood = log_likelihood
        self.log_prior = log_prior
        self.dims = dims
        self.parameters = parameters
        self.device = device

        self.periodic_parameters = periodic_parameters
        self.prior_bounds = prior_bounds
        self.bounded_to_unbounded = bounded_to_unbounded
        self.flow_matching = flow_matching
        self.flow_backend = flow_backend
        self.flow_kwargs = kwargs
        self.xp = xp

        self._flow = None

    @property
    def flow(self):
        """The normalizing flow object."""
        return self._flow

    def convert_to_samples(
        self,
        x,
        log_likelihood=None,
        log_prior=None,
        log_q=None,
        evaluate: bool = True,
    ) -> Samples:
        samples = Samples(
            x=x,
            parameters=self.parameters,
            log_likelihood=log_likelihood,
            log_prior=log_prior,
            log_q=log_q,
            xp=self.xp,
        )

        if evaluate:
            if log_prior is None:
                logger.info("Evaluating log prior")
                samples.log_prior = self.log_prior(samples)
            if log_likelihood is None:
                logger.info("Evaluating log likelihood")
                samples.log_likelihood = self.log_likelihood(samples)
            samples.compute_weights()
        return samples

    def init_flow(self):
        if self.flow_backend == "zuko":
            import array_api_compat.torch as xp
        elif self.flow_backend == "flowjax":
            import jax.numpy as xp
        data_transform = DataTransform(
            parameters=self.parameters,
            prior_bounds=self.prior_bounds,
            periodic_parameters=self.periodic_parameters,
            bounded_to_unbounded=self.bounded_to_unbounded,
            device=self.device,
            xp=xp,
        )
        self._flow = get_flow_wrapper(
            backend=self.flow_backend,
            flow_matching=self.flow_matching,
        )(dims=self.dims, data_transform=data_transform, **self.flow_kwargs)

    def fit(self, samples: Samples, **kwargs) -> dict:
        if self.xp is None:
            self.xp = samples.xp

        if self.flow is None:
            self.init_flow()

        self.training_samples = samples
        return self.flow.fit(samples.x, **kwargs)

    def sample_posterior(self, n_samples: int = 1) -> Samples:
        x, log_q = self.flow.sample_and_log_prob(n_samples)
        samples = self.convert_to_samples(x, log_q=log_q)
        logger.info("Sample summary:")
        logger.info(samples)
        return samples
