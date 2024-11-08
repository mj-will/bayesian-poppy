import jax.numpy as jnp
import jax.random as jrandom
from flowjax.train import fit_to_data

from ..base import Flow
from .utils import get_flow


class FlowJax(Flow):
    def __init__(self, dims: int, key, data_transform=None, **kwargs):
        super().__init__(dims)
        self.key = key
        self.loc = None
        self.scale = None
        self.key, subkey = jrandom.split(self.key)
        self._flow = get_flow(
            key=subkey,
            dims=self.dims,
            **kwargs,
        )
        self.data_transform = data_transform

    def fit(self, x):
        from ...history import History

        x = jnp.asarray(x)
        x_prime = self.fit_data_transform(x)
        self.key, subkey = jrandom.split(self.key)
        self._flow, losses = fit_to_data(subkey, self._flow, x_prime)
        return History(
            training_loss=list(map(lambda x: x.item(), losses["train"])),
            validation_loss=list(map(lambda x: x.item(), losses["val"])),
        )

    def sample_and_log_prob(self, n_samples: int, xp: jnp):
        self.key, subkey = jrandom.split(self.key)
        x_prime = self._flow.sample(subkey, (n_samples,))
        log_prob = self._flow.log_prob(x_prime)
        x, log_abs_det_jacobian = self.inverse_rescale(x_prime)
        return xp.asarray(x), xp.asarray(log_prob - log_abs_det_jacobian)
