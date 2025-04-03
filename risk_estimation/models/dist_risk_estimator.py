
class DistanceRiskEstimator(RiskEstimatorBase):
    APPROACH = "DIST"

    def __init__(self, name: str, dist_fun = None, thr: float = None, batch_size = 40, video_embedder = None):
        self.name = name
        self.dist_fun = dist_fun
        self.thr = thr

        self.batch_size = batch_size
        
        if video_embedder is not None:
            self.load_representation(name, video_embedder)

    def risk_to_decision(self, prob: float) -> int:
        """3: Risk --> Decision

        Args:
            prob (float): Probability of riskiness
            thr (float, optional): Threshold of decision

        Returns:
            int: Decision (1 safe or 0 risk)
        """
        return np.array(prob > self.thr, dtype=int)
    

    def training_loop(self, dataloader):
        print("No training implemented")

    def load_model(self):
        """Load hyperparameters"""
        df = pd.read_csv(f"{self.model_path}/{self.name}_{self.__class__.__name__}_hyperparam.csv")
        _, dist_fun, thr = list(df.loc[0])[0:3]
        self.dist_fun = eval(dist_fun)
        self.thr = thr

        print(f"dist_fun: {self.dist_fun}, and thr: {self.thr} loaded")


    def save_model(self):
        """Save hyperparameters"""        
        df = pd.DataFrame(np.array([[self.dist_fun.__name__, self.thr]]), columns=['dist_fun', 'thr'])
        df.to_csv(f"{self.model_path}/{self.name}_{self.__class__.__name__}_hyperparam.csv", index_label='Time')

    def normalize_dist(self, dist):
        # TODO: Generalize
        if self.dist_fun == cosine:
            # cosine dist returns values range: 0-2
            # By dividing 2, then the dist range: 0-1
            return dist / 2
        else:
            return dist

    def sample(self, x):
        """ x: observations vector, where x[:,-1] is frame number """
        ldim = self.latent_dim

        try: # to cpu if on gpu
            x = x.cpu().numpy().squeeze()
        except:
            pass
        
        if x.ndim == 1:
            x = x[None, x]
        assert x.ndim == 2

        ret_pred = []
        ret_risk = []
        for x_ in x:
            
            latent = x_[0:ldim]
            frame_normnum = x_[ldim]
            
            l = len(self.latent_images)
            repr_frame_number = int(frame_normnum * l)

            z1 = self.latent_images[repr_frame_number]

            risk_dist = self.test(z1, z2=latent)
            risk_dist = self.normalize_dist(risk_dist)

            pred = self.risk_to_decision(risk_dist)
            
            ret_pred.append(pred)
            ret_risk.append(risk_dist)

        return np.array(ret_pred), np.array(ret_risk), 0.0

    def load_representation(self, name, video_embedder):
        data = RiskEstimationDataset.load_video_data(name)
        tensor_images = data[0]
        
        if SAVE_GPU_SPACE: # save some GPU memory by encoding through batches
            dl = DataLoader(tensor_images, batch_size=100)
            out = []
            with torch.no_grad():
                for batch in dl:
                    latent_images_batch = video_embedder.model.encoder(batch)
                    out.append(latent_images_batch)
            latent_images = torch.cat(out, dim=0)
        else:
            latent_images = video_embedder.model.encoder(tensor_images)
        
        self.latent_dim = video_embedder.latent_dim
        self.tensor_images = tensor_images.cpu().detach().numpy()
        self.latent_images = latent_images.cpu().detach().numpy()

    def latent_to_risk(self, latent_image, n_image: int):
        ''' 2 '''
        latent_test = latent_image.cpu().detach().numpy().ravel()
        latent_repr = self.latent_images[n_image].cpu().detach().numpy().ravel()
        return self.test(latent_test, latent_repr)
        
    def test(self, z1, z2):
        return self.dist_fun(z1, z2)

    @staticmethod
    def make_distance(t1, t2, dist_fun):
        dist = np.zeros((len(t1), len(t2)))
        for n in range(len(t1)):
            for m in range(len(t2)):
                dist[n,m] = dist_fun(t1[n], t2[m])
        return dist

    def encode_trajectory(self, name, video_embedder):
        video_embedder.load(name=name)
        encoded_traj_1 = video_embedder.model.encoder(video_embedder.tensor_images)
        return encoded_traj_1.cpu().detach().numpy()

    def test_all_on_video_names(self, video_names, video_embedder):
        """Loads video demonstrations, safe and dangerous videos
            Comparison is w.r.t. skill_video_name skill

        Args:
            skill_video_name (_type_): _description_
            safe_names (_type_): _description_
            video_embedder (_type_): _description_
        """
        Y_pred = []
        Y_test = []
        for video_name in video_names:
            dataset_traj = RiskEstimationDataset.load_dataset([video_name], video_embedder, frame_dropping_policy=OnlyLabelledFramesDroppingPolicy,
            features=LatentObservationsRiskLabels)
            X_traj = dataset_traj.X.cpu().detach().numpy()
            y_traj = dataset_traj.Y.cpu().detach().numpy()

            Y_pred_risk, test_path_idx = self.compare_trajectories(self.latent_images, X_traj)

            print("Risk mean: ", np.median(Y_pred_risk))
    
            y_pred = self.risk_to_decision(np.array(Y_pred_risk))
            Y_pred.extend(y_pred)

            y_traj = y_traj.squeeze()
            Y_test.extend((y_traj[test_path_idx]).squeeze())
        
        assert len(Y_test) == len(Y_pred)

        return np.array(Y_test), np.array(Y_pred)
    
    def compare_trajectories(self, encoded_traj_1: np.ndarray, encoded_traj_2: np.ndarray):
        pred = []
        idxs = []
        for n, (i1, i2) in enumerate(zip(encoded_traj_1, encoded_traj_2)):
            pred.append(self.test(i1, i2))
            idxs.append(n)
        assert len(pred) == len(idxs)
        return pred, idxs
    
    def cross_test(self, t1: Iterable[float], t2: Iterable[float]):
        """Calls self.dist_fun for every combination of t1 and t2

        Args:
            t1 (Iterable[float]): 
            t2 (Iterable[float]): 

        Returns:
            float[len(t1), len(t2)]: Distance 
        """        
        dist = np.zeros((len(t1), len(t2)))
        for n in range(len(t1)):
            for m in range(len(t2)):
                dist[n, m] = self.dist_fun(t1[n], t2[m])
        return dist

class LinSearchDistanceRiskEstimator(DistanceRiskEstimator):

    def find_optimal_threshold(self, X, Y_true, thresholds):
        best_threshold = None
        best_score = 0  # Assuming higher score is better; adjust based on metric
        
        for threshold in thresholds:
            self.thr = threshold
            Y_pred, _, _ = self.sample(X)
            score = accuracy_score(Y_true.cpu().numpy(), Y_pred)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold, best_score, 0.0
    

    def training_loop(self, dataloader):
        """Searches for threshold and best-performing dist fun
        """
        self.dist_fun = cosine
        assert self.dist_fun == cosine

        X, Y_true = RiskEstimationDataset.dataloader_to_array(dataloader)
        Y_true = Y_true.squeeze()
        # cosine dist
        thresholds = np.linspace(0, 2, 50)  # 50 thresholds evenly spaced between 0 and 1
        cosine_optimal_threshold, cosine_optimal_score = self.find_optimal_threshold(X, Y_true, thresholds)

        print(f"Cosine dist")
        print(f"Optimal Threshold: {cosine_optimal_threshold}")
        print(f"Optimal Score: {cosine_optimal_score}")

        self.thr = cosine_optimal_threshold
        return

        # euclidean dist
        self.dist_fun = euclidean
        thresholds = np.linspace(0, 20, 50)  # 50 thresholds evenly spaced between 0 and 1
        euclidean_optimal_threshold, euclidean_optimal_score = self.find_optimal_threshold(X, Y_true, thresholds)

        print(f"Euclidean dist")
        print(f"Optimal Threshold: {euclidean_optimal_threshold}")
        print(f"Optimal Score: {euclidean_optimal_score}")

        if cosine_optimal_score > euclidean_optimal_score:
            print(f"Choosing Cosine")
            self.thr = cosine_optimal_threshold
            self.dist_fun = cosine
        else:
            print(f"Choosing Euclidean")
            self.thr = euclidean_optimal_threshold
            self.dist_fun = euclidean




class NMDistanceRiskEstimator(DistanceRiskEstimator):
    def sample(self, trajectories):
        """Non-Markovian sampler; Uses trajectories as batches

        Args:
            skill_video_name (_type_): _description_
            safe_names (_type_): _description_
            video_embedder (_type_): _description_
        """
        
        Y_risk = []
        Y_pred = []
        Y_idxs = []
        for trajectory in trajectories:
            y_risk, y_idxs = self.compare_trajectories(self.latent_images, trajectory)
            y_pred = self.risk_to_decision(np.array(y_risk))

            Y_risk.append(y_risk)
            Y_pred.append(y_pred)
            Y_idxs.append(y_idxs)
            
        return np.array(Y_pred, dtype=object), np.array(Y_risk, dtype=object), np.array(Y_idxs, dtype=object)
    



class MinHyperTrainDistanceRiskEstimator(DistanceRiskEstimator):
    def __init__(self, name: str, batch_size = 40, video_embedder = None):
        self.name = name
        self.batch_size = batch_size
        self.dist_fun = cosine
        self.thr = 0.5

        if video_embedder is not None:
            self.load_representation(name, video_embedder)

    def training_loop(self, dataloader, search_for_outliers: bool = True):
        """ Searches for threshold and best-performing dist fun
            Form of Gradient Descent
            Trains the logistic regression
        """

        X, y = RiskEstimationDataset.dataloader_to_array(dataloader)
        X = X.cpu().numpy().squeeze()
        y = y.cpu().numpy().squeeze()

        pred, risks, _ = self.sample(X)
        
        if not search_for_outliers:
            self.thr = np.min(risks[y == 1]) - 1e-4
            print(f"Selected minimum threshold: {self.thr}")
            return

        else:
            outlier_thr = 0.02
            risky_array = risks[y == 1]
            max_outliers_tested = 5
            mins = []
            for i in range(max_outliers_tested):
                mins.append(np.min(risky_array))
                risky_array[np.argmin(risky_array)] = np.inf

                if i==0: continue # cannot compare: min[0] == min[-1]

                is_outlier = abs(mins[-1] - mins[0]) > outlier_thr
                if not is_outlier:
                    break
            
            print(f"Selected threshold, considering outliers: {mins[-1]}, mins {mins}")
            self.thr = mins[-1]
            plot_threshold_labelled(risks, y)

class NMMinHyperTrainDistanceRiskEstimator(NMDistanceRiskEstimator, MinHyperTrainDistanceRiskEstimator):
    pass


class LRHyperTrainDistanceRiskEstimator(DistanceRiskEstimator):
    def __init__(self, name: str, batch_size = 40, video_embedder = None):
        self.name = name
        self.batch_size = batch_size
        self.dist_fun = cosine
        
        if video_embedder is not None:
            self.load_representation(name, video_embedder)
    
    def compute_cost(self, X, y, y_pred):
        assert len(y) == len(y_pred)
        # Binary cross-entropy cost
        m = len(y)
        epsilon = 1e-5  # to prevent log(0)
        cost = -np.sum(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon)) / m
        return cost

    def sample_with_thr(self, x, thr):
        self.thr = thr
        ret, ret_risk, _ = self.sample(x)
        return ret, ret_risk

    def compute_gradient_numerical(self, X, y, weight, eps=5e-2):
        # Numerical gradient approximation

        grad_plus,_ = self.sample_with_thr(X, weight + eps)
        grad_minus,_ = self.sample_with_thr(X, weight - eps)
        numerical_gradient = (self.compute_cost(X, y, grad_plus) - self.compute_cost(X, y, grad_minus)) / (2 * eps)
        return numerical_gradient

    def training_loop(self, dataloader):
        """ Searches for threshold and best-performing dist fun
            Form of Gradient Descent
            Trains the logistic regression
        """

        X, y = RiskEstimationDataset.dataloader_to_array(dataloader)
        X = X.cpu().numpy().squeeze()
        y = y.cpu().numpy().squeeze()
        
        threshold = 1.0
        self.learning_rate = 0.001
        cost_history = []

        last_grad = -1
        for i in range(self.train_epoch):
            
            gradient = self.compute_gradient_numerical(X, y, threshold)
            
            if abs(gradient) < 0.001: # getting out of local minimum
                threshold -= self.learning_rate * last_grad
            else:
                threshold -= self.learning_rate * gradient
                last_grad = gradient

            y_pred, _ = self.sample_with_thr(X, threshold)

            cost = np.sum(np.abs(y_pred - y))

            cost_history.append(cost)
            
            if i % 100 == 0:
                print(f"Acc: {(y_pred == y).mean()}")
                print(f"Gradient: {gradient}")
                print(f"Cost at iteration {i}: {cost}")
                print(f"Threshold {threshold}")

        self.thr = threshold
        print(f"Threshold is {self.thr}")

  

