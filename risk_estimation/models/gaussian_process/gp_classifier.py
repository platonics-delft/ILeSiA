import gpytorch
import torch

# We will use the simplest form of GP model, exact inference
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, ard=False):
        self.train_x = train_x
        self.train_y = train_y
        if ard:
            ard_num_dim=train_x.size(-1)
        else:
            ard_num_dim=None
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        # self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = gpytorch.means.ZeroMean()  
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dim))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPModelPrepro(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, ard=False, l=8):
        self.train_x = train_x
        self.train_y = train_y
        super(GPModelPrepro, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()  
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dim))
        self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=l)

        self.prepro = torch.nn.Linear(train_x.size(-1), l).cuda()
        self.prepro_norm = torch.nn.BatchNorm1d(train_x.size(-1))

    def forward(self, x):
        x = self.prepro_norm(x)
        x = self.prepro(x)  # Dimensionality Reduction
        # x = (x - x.mean()) / x.std()
        # x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)
        
        # components = kernel_pca(x, n_components=3)
        # x = project_data(x, components)

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPModelPreproFeatureSkip(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, ard=False, l=5, n_features_to_skip=1):
        self.train_x = train_x
        self.train_y = train_y
        self.nskip = n_features_to_skip
        super(GPModelPreproFeatureSkip, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()  
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dim))
        self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=l+self.nskip)

        self.prepro = torch.nn.Linear(train_x.size(-1)-self.nskip, l).cuda()


    def forward(self, x):
        x1 = self.prepro(x[:,:-self.nskip])  # Dimensionality Reduction
        x1 = (x1 - x1.mean(dim=0, keepdim=True)) / (x1.std(dim=0, keepdim=True) + 1e-6)

        x2 = torch.cat([x1,x[:,-self.nskip:]], dim=1)
        mean_x = self.mean_module(x2)
        covar_x = self.covar_module(x2)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



def rbf_kernel(X, gamma=None):
    if gamma is None:
        gamma = 1.0 / X.size(1)  # default gamma value: 1/d where d is the number of features

    sq_dists = torch.cdist(X, X, p=2)  # Compute the pairwise Euclidean distance
    return torch.exp(-gamma * sq_dists)  # Apply the RBF kernel

def center_kernel(K):
    N = K.size(0)
    one_n = torch.ones(N, N) / N
    return K - one_n @ K - K @ one_n + one_n @ K @ one_n

def kernel_pca(X, n_components=2):
    K = rbf_kernel(X)
    K = center_kernel(K)
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(K)
    # Sort eigenvalues: largest to smallest
    idx = eigenvalues.argsort(descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select the top 'n_components' eigenvectors
    return eigenvectors[:, :n_components]

def project_data(X, components):
    K = rbf_kernel(X)
    return torch.mm(K, components)
