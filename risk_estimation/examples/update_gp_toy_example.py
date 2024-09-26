

import time
from tqdm import tqdm
import torch
import gpytorch
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import UnwhitenedVariationalStrategy
# slip the data into training and testing set
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles, make_classification, make_moons

import numpy as np
from matplotlib import pyplot as plt

# We will use the simplest form of GP model, exact inference
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, ard=False):

        if ard:
            ard_num_dim=train_x.size(-1)
        else:
            ard_num_dim=None
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        # self.mean_module = gpytorch.means.ConstantMean(with_contratint)
        self.mean_module = gpytorch.means.ZeroMean()  
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dim))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


        
#%%
class Classifier():

    def create_model(self):        
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        print("Has  analytical likelihood:")
        print(self.likelihood.has_analytic_marginal)

        self.model = GPModel( self.X, self.y, self.likelihood, ard=self.ard)
        # self.model.mean_module= the_old_one

    def move_model_to_cuda(self):
        self.model=self.model.cuda()
        self.likelihood=self.likelihood.cuda()
        self.X=self.X.cuda()
        self.y=self.y.cuda()
        self.model=self.model.double()
        self.likelihood=self.likelihood.double()

    def train(self, X, y, num_epochs=3, ard=True):
        #pick the inducing from the training set 
        self.ard=ard  
        X=X.astype(float)
        y=y.astype(float)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.create_model()
        self.move_model_to_cuda()
        self.train_loop(num_epochs)
        
    def train_loop(self, num_epochs):
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
        ], lr=0.01)

        # Our loss object. We're using the VariationalELBO
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        train_dataset = TensorDataset(self.X, self.y)
        self.train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

        # epochs_iter = tqdm.notebook.tqdm(range(num_epochs), desc="Epoch")
        epochs_iter = tqdm(range(num_epochs))
        for i in epochs_iter:
            optimizer.zero_grad()
            output = self.model(self.X)
            loss = -mll(output, self.y)
            loss.backward()
            optimizer.step()
        

    
    def predict(self, X):
        self.model.eval()
        self.likelihood.eval()
        X=X.astype(float)
        X = torch.from_numpy(X).cuda().double()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(X))
            return observed_pred.mean.cpu().numpy()
        


def plot_it(classifier, X_train, X_test, y_train, y_test):
    ax = plt.subplot(1, 1 , 1)
    # make prediction on a grip and color it 
    x_min, x_max = X_train[:, 0].min()-3, X_train[:, 0].max() +3 
    y_min, y_max = X_train[:, 1].min()-3, X_train[:, 1].max() +3

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    Z[Z>0.5]=1
    Z[Z<=0.5]=0

    Z = Z.reshape(xx.shape)

    cs = ax.contourf(xx, yy, Z)
    ax.contour(cs, colors='k', alpha= 0.2)

    ax.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, edgecolors="k"
        )
    # Plot the testing points
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        # cmap=cm_bright,
        edgecolors="k",
        alpha=0.6,
    )
    plt.title("Classification")
    plt.tight_layout()
    plt.show()

def make_acc(y_pred, y_test):
    y_pred[y_pred>0.5]=1
    y_pred[y_pred<=0.5]=0
    accuracy=(y_pred==y_test).sum()/len(y_test)
    print('accuracy:',accuracy)
#%% Initial train
print("Initial train")

X, y  = make_moons(noise=0.3, random_state=0, n_samples=400)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = Classifier()
t1 = time.perf_counter()
classifier.train(X=X_train, y=y_train, num_epochs=100) 
print(f"Train Time: {time.perf_counter()-t1}")
print(f"Samples {len(X_train)}")

y_pred=classifier.predict(X_test)
make_acc(y_pred, y_test)
plot_it(classifier, X_train, X_test, y_train, y_test)

#%% Add some data
print("Update train")

X_add = np.array([np.random.random(10) + 2, np.random.random(10) + 2]).T
y_add = np.ones(10)
X_train = np.concatenate((X_train, X_add))
y_train = np.concatenate((y_train, y_add))

t1 = time.perf_counter()
classifier.train(X=X_train, y=y_train, num_epochs=100) 
print(f"Train Time: {time.perf_counter()-t1}")
print(f"Samples {len(X_train)}")

y_pred=classifier.predict(X_test)
make_acc(y_pred, y_test)
plot_it(classifier, X_train, X_test, y_train, y_test)

#%% Using get fantasy model func
print("Fantasy train")

t1 = time.perf_counter()
new_model = classifier.model.get_fantasy_model(
    torch.tensor(X_train).cuda(),
    torch.tensor(y_train).cuda()
)
classifier.train_loop(num_epochs=100) 
print(f"Train Time: {time.perf_counter()-t1}")
print(f"Samples {len(X_train)}")

y_pred=classifier.predict(X_test)
make_acc(y_pred, y_test)
plot_it(classifier, X_train, X_test, y_train, y_test)

print("Lengthscale: ")        
print(classifier.model.covar_module.base_kernel.lengthscale)
print("Outputscale: ")
print(torch.sqrt(classifier.model.covar_module.outputscale))


