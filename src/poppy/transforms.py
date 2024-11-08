import copy
import logging
import math

from scipy.special import erf, erfinv

logger = logging.getLogger(__name__)


class DataTransform:
    def __init__(
        self,
        parameters: list[int],
        periodic_parameters: list[int],
        prior_bounds: list[tuple[float, float]],
        bounded_to_unbounded: bool = True,
        device=None,
        xp: None = None,
    ):
        if prior_bounds is None:
            logger.warning(
                "Missing prior bounds, some transforms may not be applied."
            )
        if periodic_parameters and not prior_bounds:
            raise ValueError(
                "Must specify prior bounds to use periodic parameters."
            )
        self.parameters = parameters
        if periodic_parameters:
            logger.warning("Periodic parameters are not implemented yet.")
        self.periodic_parameters = []
        self.bounded_to_unbounded = bounded_to_unbounded

        self.xp = xp
        self.device = device

        if prior_bounds is None:
            self.prior_bounds = None
            self.bounded_parameters = None
            lower_bounds = None
            upper_bounds = None
        else:
            logger.info(f"Prior bounds: {prior_bounds}")
            self.prior_bounds = {
                k: self.xp.asarray(v) for k, v in prior_bounds.items()
            }
            if bounded_to_unbounded:
                self.bounded_parameters = [
                    p
                    for p in parameters
                    if self.xp.isfinite(self.prior_bounds[p]).all()
                    and p not in self.periodic_parameters
                ]
            else:
                self.bounded_parameters = None
            lower_bounds = self.xp.asarray(
                [self.prior_bounds[p][0] for p in parameters]
            )
            upper_bounds = self.xp.asarray(
                [self.prior_bounds[p][1] for p in parameters]
            )

        if self.periodic_parameters:
            logger.info(f"Periodic parameters: {self.periodic_parameters}")
            self.periodic_mask = self.xp.asarray(
                [p in self.periodic_parameters for p in parameters], dtype=bool
            )
            self.periodic_transform = PeriodicTransform(
                lower=lower_bounds[self.periodic_mask],
                upper=upper_bounds[self.periodic_mask],
                xp=self.xp,
            )
        if self.bounded_parameters:
            logger.info(f"Bounded parameters: {self.bounded_parameters}")
            self.bounded_mask = self.xp.asarray(
                [p in self.bounded_parameters for p in parameters], dtype=bool
            )
            self.bounded_transform = ProbitTransform(
                lower=lower_bounds[self.bounded_mask],
                upper=upper_bounds[self.bounded_mask],
                xp=self.xp,
            )
        logger.info(f"Affine transform applied to: {self.parameters}")
        self.affine_transform = AffineTransform(xp=self.xp)

    def fit(self, x):
        x = copy.copy(x)
        if self.periodic_parameters:
            logger.debug(
                f"Fitting periodic transform to parameters: {self.periodic_parameters}"
            )
            x[:, self.periodic_mask] = self.periodic_transform.fit(
                x[:, self.periodic_mask]
            )
        if self.bounded_parameters:
            logger.debug(
                f"Fitting bounded transform to parameters: {self.bounded_parameters}"
            )
            x[:, self.bounded_mask] = self.bounded_transform.fit(
                x[:, self.bounded_mask]
            )
        return self.affine_transform.fit(x)

    def forward(self, x):
        x = copy.copy(x)
        log_abs_det_jacobian = self.xp.zeros(len(x), device=self.device)
        if self.periodic_parameters:
            x[:, self.periodic_mask], log_j_periodic = (
                self.periodic_transform.forward(x[:, self.periodic_mask])
            )
            log_abs_det_jacobian += log_j_periodic

        if self.bounded_parameters:
            x[:, self.bounded_mask], log_j_bounded = (
                self.bounded_transform.forward(x[:, self.bounded_mask])
            )
            log_abs_det_jacobian += log_j_bounded

        x, log_j_affine = self.affine_transform.forward(x)
        log_abs_det_jacobian += log_j_affine
        return x, log_abs_det_jacobian

    def inverse(self, x):
        x = copy.copy(x)
        log_abs_det_jacobian = self.xp.zeros(len(x), device=self.device)
        x, log_j_affine = self.affine_transform.inverse(x)
        log_abs_det_jacobian += log_j_affine

        if self.bounded_parameters:
            x[:, self.bounded_mask], log_j_bounded = (
                self.bounded_transform.inverse(x[:, self.bounded_mask])
            )
            log_abs_det_jacobian += log_j_bounded

        if self.periodic_parameters:
            x[:, self.periodic_mask], log_j_periodic = (
                self.periodic_transform.inverse(x[:, self.periodic_mask])
            )
            log_abs_det_jacobian += log_j_periodic

        return x, log_abs_det_jacobian


class PeriodicTransform(DataTransform):
    name: str = "periodic"
    requires_prior_bounds: bool = True

    def __init__(self, lower, upper, xp):
        self.lower = lower
        self.upper = upper
        self._width = upper - lower
        self._shift = None
        self.xp = xp

    def fit(self, x):
        return self.forward(x)[0]

    def forward(self, x):
        y = self.lower + (x - self.lower) % self._width
        return y, self.xp.zeros(y.shape[0], device=y.device)

    def inverse(self, y):
        x = self.lower + (y - self.lower) % self._width
        return x, self.xp.zeros(x.shape[0], device=x.device)


class ProbitTransform(DataTransform):
    name: str = "probit"
    requires_prior_bounds: bool = True

    def __init__(self, lower, upper, xp, eps=1e-6):
        self.lower = lower
        self.upper = upper
        self._scale_log_abs_det_jacobian = -xp.log(upper - lower).sum()
        self.eps = eps
        self.xp = xp

    def fit(self, x):
        return self.forward(x)[0]

    def forward(self, x):
        y = (x - self.lower) / (self.upper - self.lower)
        y = self.xp.clip(y, self.eps, 1.0 - self.eps)
        y = erfinv(2 * y - 1) * math.sqrt(2)
        log_abs_det_jacobian = (
            0.5 * (math.log(2 * math.pi) + y**2).sum(-1)
            + self._scale_log_abs_det_jacobian
        )
        return y, log_abs_det_jacobian

    def inverse(self, y):
        log_abs_det_jacobian = (
            -(0.5 * (math.log(2 * math.pi) + y**2)).sum(-1)
            - self._scale_log_abs_det_jacobian
        )
        x = 0.5 * (1 + erf(y / math.sqrt(2)))
        x = (self.upper - self.lower) * x + self.lower
        return x, log_abs_det_jacobian


class AffineTransform(DataTransform):
    name: str = "affine"
    requires_prior_bounds: bool = False

    def __init__(self, xp):
        self._mean = None
        self._std = None
        self.xp = xp

    def fit(self, x):
        self._mean = x.mean(0)
        self._std = x.std(0)
        self.log_abs_det_jacobian = -self.xp.log(self.xp.abs(self._std)).sum()
        return self.forward(x)[0]

    def forward(self, x):
        y = (x - self._mean) / self._std
        return y, self.log_abs_det_jacobian * self.xp.ones(
            y.shape[0], device=y.device
        )

    def inverse(self, y):
        x = y * self._std + self._mean
        return x, -self.log_abs_det_jacobian * self.xp.ones(
            y.shape[0], device=y.device
        )
