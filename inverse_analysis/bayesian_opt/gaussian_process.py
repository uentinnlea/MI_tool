import torch
import numpy as np
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize
from botorch import fit_gpytorch_mll
from gpytorch.kernels import LinearKernel, RBFKernel, MaternKernel
from sklearn.base import BaseEstimator


class GaussianProcessRegressor(BaseEstimator):
    def __init__(self, model=None, kernel_name=None):
        self.kernel_name_list = [LinearKernel.__name__, RBFKernel.__name__, MaternKernel.__name__]
        self.model = model

        if kernel_name == MaternKernel.__name__:
            self.kernel = MaternKernel()
        elif kernel_name == RBFKernel.__name__:
            self.kernel = RBFKernel()
        elif kernel_name == LinearKernel.__name__:
            self.kernel = LinearKernel()
        else:
            raise ValueError("Invalid kernel_name")

    def fit(self, X:np.ndarray, y: np.ndarray):
        gp_model = SingleTaskGP(
            torch.from_numpy(X),
            torch.from_numpy(y).unsqueeze(-1),
            outcome_transform=Standardize(m=1),
            input_transform=Normalize(d=X.shape[1]),
            covar_module=self.kernel)

        self.model = gp_model
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        return self
    
    def posterior_result(self, X: np.ndarray):
        posterior = self.model.posterior(torch.from_numpy(X))
        mean = posterior.mean.to("cpu").detach().numpy().copy()
        variance = posterior.variance.to("cpu").detach().numpy().copy()
        return mean, variance
    
    