import torch
import tqdm
import zuko
from array_api_compat import is_torch_array
from array_api_compat import torch as torch_api

from ..base import Flow


class BaseTorchFlow(Flow):
    def __init__(
        self,
        dims: int,
        seed: int = 1234,
        device: str = "cpu",
        data_transform=None,
    ):
        super().__init__(dims, data_transform=data_transform)
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.loc = None
        self.scale = None

    def fit(self, x):
        raise NotImplementedError()


class ZukoFlow(BaseTorchFlow):
    def __init__(
        self,
        dims,
        flow_class: str = "MAF",
        data_transform=None,
        seed=1234,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            dims, data_transform=data_transform, seed=seed, device=device
        )
        FlowClass = getattr(zuko.flows, flow_class)
        self._flow = FlowClass(self.dims, 0, **kwargs)
        self._flow.compile()

    def loss_fn(self, x):
        return -self._flow().log_prob(x).mean()

    def fit(
        self,
        x,
        n_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 500,
        validation_fraction: float = 0.2,
        lr_annealing: bool = False,
    ):
        from ...history import History

        if not is_torch_array(x):
            x = torch.tensor(
                x, dtype=torch.get_default_dtype(), device=self.device
            )
        else:
            x = torch.clone(x)
        x_prime = self.fit_data_transform(x)
        indices = torch.randperm(x_prime.shape[0])
        x_prime = x_prime[indices, ...]

        n = x_prime.shape[0]
        x_train = torch.as_tensor(
            x_prime[: -int(validation_fraction * n)],
            dtype=torch.get_default_dtype(),
            device=self.device,
        )
        x_val = torch.as_tensor(
            x_prime[-int(validation_fraction * n) :],
            dtype=torch.get_default_dtype(),
            device=self.device,
        )

        dataset = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train),
            shuffle=True,
            batch_size=batch_size,
        )
        val_dataset = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_val),
            shuffle=False,
            batch_size=batch_size,
        )

        # Train to maximize the log-likelihood
        optimizer = torch.optim.Adam(self._flow.parameters(), lr=lr)
        if lr_annealing:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, n_epochs
            )
        history = History()

        for _ in tqdm.tqdm(range(n_epochs)):
            self._flow.train()
            loss_epoch = 0.0
            for (x_batch,) in dataset:
                loss = self.loss_fn(x_batch)
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(flow.parameters(), 2.0)
                optimizer.step()
                loss_epoch += loss.item()
            if lr_annealing:
                scheduler.step()
            history.training_loss.append(loss_epoch / len(dataset))
            self._flow.eval()
            val_loss = 0.0
            for (x_batch,) in val_dataset:
                with torch.no_grad():
                    val_loss += self.loss_fn(x_batch).item()
            history.validation_loss.append(val_loss / len(val_dataset))
        return history

    def sample_and_log_prob(self, n_samples: int, xp=torch_api):
        with torch.no_grad():
            x_prime, log_prob = self._flow().rsample_and_log_prob((n_samples,))
        x, log_abs_det_jacobian = self.inverse_rescale(x_prime)
        return xp.asarray(x), xp.asarray(log_prob - log_abs_det_jacobian)

    def log_prob(self, x, xp=torch_api):
        x = torch.tensor(
            x, dtype=torch.get_default_dtype(), device=self.device
        )
        x_prime, log_abs_det_jacobian = self.rescale(x)
        return xp.asarray(
            self._flow().log_prob(x_prime) + log_abs_det_jacobian
        )


class ZukoFlowMatching(ZukoFlow):
    def __init__(
        self,
        dims,
        data_transform=None,
        seed=1234,
        device="cpu",
        eta: float = 1e-3,
        **kwargs,
    ):
        kwargs.setdefault("hidden_features", 4 * [100])
        super().__init__(
            dims,
            seed=seed,
            device=device,
            data_transform=data_transform,
            flow_class="CNF",
        )
        self.eta = eta

    def loss_fn(self, theta: torch.Tensor):
        t = torch.rand(
            theta.shape[:-1], dtype=theta.dtype, device=theta.device
        )
        t_ = t[..., None]
        eps = torch.randn_like(theta)
        theta_prime = (1 - t_) * theta + (t_ + self.eta) * eps
        v = eps - theta
        return (self._flow.transform.f(t, theta_prime) - v).square().mean()
