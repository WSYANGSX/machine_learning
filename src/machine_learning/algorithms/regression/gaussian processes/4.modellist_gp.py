import math
import torch
import gpytorch
from matplotlib import pyplot as plt


# train data
train_x1 = torch.linspace(0, 0.95, 50) + 0.05 * torch.rand(50)
train_x2 = torch.linspace(0, 0.95, 25) + 0.05 * torch.rand(25)

train_y1 = torch.sin(train_x1 * (math.pi * 2) + 0.2 * torch.randn_like(train_x1))
train_y2 = torch.sin(train_x2 * (math.pi * 2) + 0.2 * torch.randn_like(train_x2))


# setup model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super().__init__(train_inputs, train_targets, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


likelihood1 = gpytorch.likelihoods.GaussianLikelihood()
model1 = ExactGPModel(train_x1, train_y1, likelihood1)

likelihood2 = gpytorch.likelihoods.GaussianLikelihood()
model2 = ExactGPModel(train_x2, train_y2, likelihood2)

model = gpytorch.models.IndependentModelList(model1, model2)
likelihood = gpytorch.likelihoods.LikelihoodList(model1.likelihood, model2.likelihood)


from gpytorch.mlls import SumMarginalLogLikelihood

mll = SumMarginalLogLikelihood(likelihood, model)


# train the model
import os

somke_test = "CI" in os.environ
training_iter = 2 if somke_test else 50


model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for i in range(training_iter):
    optimizer.zero_grad()
    output = model(*model.train_inputs)
    loss = -mll(output, model.train_targets)
    loss.backward()
    print("Iter %d/%d - Loss: %.3f" % (i + 1, training_iter, loss.item()))
    optimizer.step()

# Set into eval mode
model.eval()
likelihood.eval()

# Initialize plots
f, axs = plt.subplots(1, 2, figsize=(8, 3))

# Make predictions (use the same test points)
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.linspace(0, 1, 51)
    # This contains predictions for both outcomes as a list
    predictions = likelihood(*model(test_x, test_x))

for submodel, prediction, ax in zip(model.models, predictions, axs):
    mean = prediction.mean
    lower, upper = prediction.confidence_region()

    tr_x = submodel.train_inputs[0].detach().numpy()
    tr_y = submodel.train_targets.detach().numpy()

    # Plot training data as black stars
    ax.plot(tr_x, tr_y, "k*")
    # Predictive mean as blue line
    ax.plot(test_x.numpy(), mean.numpy(), "b")
    # Shade in confidence
    ax.fill_between(
        test_x.numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5
    )
    ax.set_ylim([-3, 3])
    ax.legend(["Observed Data", "Mean", "Confidence"])
    ax.set_title("Observed Values (Likelihood)")

plt.show()
