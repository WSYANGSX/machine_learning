import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from torch.distributions import Distribution


# training data
train_x1 = torch.rand(2000)
train_x2 = torch.rand(2000)

train_y1 = torch.sin(train_x1 * (2 * math.pi)) + torch.randn(train_x1.size()) * 0.2
train_y2 = torch.cos(train_x2 * (2 * math.pi)) + torch.randn(train_x2.size()) * 0.2


# setup a hadamard multitask model
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super().__init__(train_inputs, train_targets, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()

        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=2, rank=1)

    def forward(self, x, i):
        mean_x = self.mean_module(x)

        covar_x = self.covar_module(x)

        covar_i = self.covar_module(i)

        covar = covar_x.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)


likelihood = gpytorch.likelihoods.GaussianLikelihood()

train_i_task1 = torch.full((train_x1.shape[0], 1), dtype=torch.long, fill_value=0)
train_i_task2 = torch.full((train_x2.shape[0], 1), dtype=torch.long, fill_value=1)

full_train_x = torch.cat([train_x1, train_x2])
full_train_i = torch.cat([train_i_task1, train_i_task2])
full_train_y = torch.cat([train_y1, train_y2])

model = MultitaskGPModel((full_train_x, full_train_i), full_train_y, likelihood)


import os

smoke_test = "CI" in os.environ
training_iterations = 2 if smoke_test else 50


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.1
)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(full_train_x, full_train_i)
    loss = -mll(output, full_train_y)
    loss.backward()
    print("Iter %d/50 - Loss: %.3f" % (i + 1, loss.item()))
    optimizer.step()
    
# Set into eval mode
model.eval()
likelihood.eval()

# Initialize plots
f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))

# Test points every 0.02 in [0,1]
test_x = torch.linspace(0, 1, 51)
test_i_task1 = torch.full((test_x.shape[0], 1), dtype=torch.long, fill_value=0)
test_i_task2 = torch.full((test_x.shape[0], 1), dtype=torch.long, fill_value=1)

# Make predictions - one task at a time
# We control the task we cae about using the indices

# The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
# See https://arxiv.org/abs/1803.06058
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred_y1 = likelihood(model(test_x, test_i_task1))
    observed_pred_y2 = likelihood(model(test_x, test_i_task2))


# Define plotting function
def ax_plot(ax, train_y, train_x, rand_var, title):
    # Get lower and upper confidence bounds
    lower, upper = rand_var.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x.detach().numpy(), train_y.detach().numpy(), "k*")
    # Predictive mean as blue line
    ax.plot(test_x.detach().numpy(), rand_var.mean.detach().numpy(), "b")
    # Shade in confidence
    ax.fill_between(
        test_x.detach().numpy(),
        lower.detach().numpy(),
        upper.detach().numpy(),
        alpha=0.5,
    )
    ax.set_ylim([-3, 3])
    ax.legend(["Observed Data", "Mean", "Confidence"])
    ax.set_title(title)


# Plot both tasks
ax_plot(y1_ax, train_y1, train_x1, observed_pred_y1, "Observed Values (Likelihood)")
ax_plot(y2_ax, train_y2, train_x2, observed_pred_y2, "Observed Values (Likelihood)")

plt.show()