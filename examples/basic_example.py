"""
This examples demonstrates how to use Poppy to fit a flow to a simple Gaussian
likelihood with a uniform prior.
"""

import math

import torch
import torch.distributions

from poppy import Poppy
from poppy.samples import Samples
from poppy.utils import configure_logger

# Configure the logger
configure_logger("INFO")

# Number of dimensions
dims = 16


# Define the log likelihood and log prior
def log_likelihood(samples: Samples) -> torch.Tensor:
    # The log likelihood must accept a Samples object
    # The samples object contains the samples in the attribute samples.x
    return torch.distributions.Normal(2, 1).log_prob(samples.x).sum(axis=-1)


def log_prior(samples: Samples) -> torch.Tensor:
    return (
        torch.distributions.Uniform(-10, 10).log_prob(samples.x).sum(axis=-1)
    )


# True evidence is analytic for a Gaussian likelihood and uniform prior
true_log_evidence = -dims * math.log(20)

# Generate some initial samples
initial_samples = Samples(1.0 * torch.randn(5000, dims) + 2.0)
# Define the parameters and prior bounds
parameters = [f"x_{i}" for i in range(dims)]
prior_bounds = {p: [-10, 10] for p in parameters}

# Define the poppy object
poppy = Poppy(
    log_likelihood=log_likelihood,
    log_prior=log_prior,
    dims=dims,
    parameters=parameters,
    prior_bounds=prior_bounds,
    flow_matching=False,
    flow_backend="zuko",
)

# Fit the flow to the initial samples
history = poppy.fit(
    initial_samples,
    n_epochs=50,
    lr_annealing=True,
)
# Plot the loss
fig = history.plot_loss()
fig.savefig("loss.png")

# Produce samples from the posterior
samples = poppy.sample_posterior(2000)

# Produce a corner plot showing the samples
corner_kwargs = dict(
    density=True,
    bins=30,
    color="C0",
    hist_kwargs=dict(density=True, color="C0"),
)
fig = None
# Plot the initial samples
fig = poppy.training_samples.plot_corner(**corner_kwargs)
# Plot the samples without the weights
corner_kwargs["color"] = "lightgrey"
corner_kwargs["hist_kwargs"]["color"] = "lightgrey"
fig = samples.plot_corner(fig=fig, include_weights=False, **corner_kwargs)
# Plot the samples with the weights
corner_kwargs["color"] = "C1"
corner_kwargs["hist_kwargs"]["color"] = "C1"
fig = samples.plot_corner(fig=fig, **corner_kwargs)
fig.savefig("comparison.png")
