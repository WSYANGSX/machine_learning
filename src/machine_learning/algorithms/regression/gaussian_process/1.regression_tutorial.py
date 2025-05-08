import torch
import gpytorch
from matplotlib import pyplot as plt


# training data
# x =[0,1], y = sin(2*pi*x) + epsilon, epsilon~N(0, 0.04)
train_x = torch.linspace(0, 1, 100)
train_y = torch.sin(2 * torch.pi * train_x) + torch.randn_like(train_x) * torch.sqrt(
    torch.scalar_tensor(0.04)
)
print("train x:", train_x)
print("train y:", train_y)
print(train_x.dtype)
print(train_y.dtype)


# setup the model
# 1.GP Model
# 2.Likelihood-gpytorch.likelihoods.GaussianLikelihood：同方差噪声模型
# 3.Mean（the prior mean of the GP）
# 4.Kernel（the prior covariance of the GP）
# 5.MultivariateNormal
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super().__init__(train_inputs, train_targets, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)


# train the model use MLE
# 1.Zero all parameter gradients
# 2.Call the model and compute the loss
# 3.Call backward on the loss to fill in gradients
# 4.Take a step on the optimizer

import os

smoke_test = "CI" in os.environ
training_iter = 2 if smoke_test else 50

# 设置训练模式
model.train()
likelihood.train()

# 设置优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    # zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    print(
        "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f"
        % (
            i + 1,
            training_iter,
            loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item(),
        )
    )
    optimizer.step()


# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad():
    test_x = torch.linspace(0, 1, 51)
    observed_pred = likelihood(model(test_x))


with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y.numpy(), "k*")
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), "b")
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(["Observed Data", "Mean", "Confidence"])

    plt.show()
