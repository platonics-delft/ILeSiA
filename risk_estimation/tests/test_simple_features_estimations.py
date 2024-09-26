from typing import Iterable
from sklearn.metrics import confusion_matrix
from risk_estimation.models.risk_estimation.frame_dropping import NoFrameDroppingPolicy, OnlyLabelledFramesDroppingPolicy
from risk_estimation.models.risk_estimation.risk_dataloader import RiskEstimationDataset
from risk_estimation.models.risk_estimation.risk_feature_extractor import LatentObservationsRiskLabels, StampedLatentObservationsRiskLabels
from risk_estimation.models.risk_estimator import (
    DistanceRiskEstimator,
    GPRiskEstimator,
    LRHyperTrainDistanceRiskEstimator,
    LRRiskEstimator,
    LinSearchDistanceRiskEstimator,
    MinHyperTrainDistanceRiskEstimator,
    NMDistanceRiskEstimatorDTW,
    MLPRiskEstimator,
)
import video_embedding
from video_embedding.models.video_embedder import VideoEmbedder
import risk_estimation
from scipy.spatial.distance import cosine, euclidean
from video_embedding.utils import all_trial_names, number_of_saved_trials, set_session

def get_risk_estimator(skill_name, approach, features_cls, video_latent_dim, video_embedder):

    print("approach", approach)

    if approach == 'MLP':
        return MLPRiskEstimator(name=skill_name, 
            xdim=features_cls.xdim(video_latent_dim),
            batch_size=video_embedder.batch_size,
            arch="A4",
        )
    elif approach == 'DistLS':
        return LinSearchDistanceRiskEstimator(name=skill_name,
            dist_fun = cosine, # custom
            thr = 0.5, # custom
            video_embedder = video_embedder, # load repre
        )  
    elif approach == 'GP':
        return GPRiskEstimator(name=skill_name,
            thr = 0.5,
        )  
    elif approach == 'LR':
        return LRRiskEstimator(name=skill_name,
            xdim=features_cls.xdim(video_latent_dim),
        )
    elif approach == 'DistLR':
        return LRHyperTrainDistanceRiskEstimator(
            name=skill_name,
            video_embedder=video_embedder,
        )  
    elif approach == 'DistMin':
        return MinHyperTrainDistanceRiskEstimator(
            name=skill_name,
            video_embedder=video_embedder,
        )
    else: raise Exception()

def evaluate_hyperparams(
        skill_name: str,
        approach: str, 
        video_names: Iterable[str],
        latent_dim: int,
        framedrop,
        features,
        train: bool = False,
    ):

    # Video embedder encodes skill from video
    video_embedder = VideoEmbedder(latent_dim)
    video_embedder.load(skill_name)  # load data

    # video_embedder.create_video(path=risk_estimation.path + "/videos/")  # train ae
    video_embedder.load_model(
        path=video_embedding.path + "/saved_models/"
    )  # OR load model

    risk_estimator = get_risk_estimator(skill_name, approach, features, latent_dim, video_embedder)
    # risk_estimator = MLPRiskEstimator(
    #     skill_name,
    #     features.xdim(latent_dim),
    #     video_embedder.batch_size,
    # )

    # Risk Estimator loads data for training
    train_dl, test_dl = RiskEstimationDataset.load(
        video_names=video_names,
        video_embedder=video_embedder,
        batch_size=video_embedder.batch_size,
        frame_dropping_policy=framedrop,
        features=features,
    )

    risk_estimator.training_loop(
        train_dl, num_epochs=2500, patience=10000
    )  # Train Risk Aware module
    risk_estimator.save_model(
        path=risk_estimation.path + "/saved_models/"
    )

    X, Y = RiskEstimationDataset.dataloader_to_array(test_dl)
    
    Y = Y.cpu().numpy().squeeze()
    Y_pred, _ = risk_estimator.sample(X)

    acc = 100 * (Y == Y_pred).mean()
    print(acc)
    return acc

def test_E4():
    set_session("test_session")
    skill_name='peg_door'
    acc = evaluate_hyperparams(skill_name,
        approach="MLP",
        video_names = [f"{skill_name}_trial_{i}" for i in range(number_of_saved_trials(skill_name))],
        latent_dim=8,
        framedrop=OnlyLabelledFramesDroppingPolicy,
        features=StampedLatentObservationsRiskLabels,
    )
    assert acc > 92

def test_E5():
    set_session("test_session")
    skill_name='peg_pick'
    acc = evaluate_hyperparams(skill_name,
        approach="MLP",
        video_names = [f"{skill_name}_trial_{i}" for i in range(number_of_saved_trials(skill_name))],
        latent_dim=8,
        framedrop=OnlyLabelledFramesDroppingPolicy,
        features=StampedLatentObservationsRiskLabels,
        train=False,
    )
    if acc < 92:
        acc = evaluate_hyperparams(skill_name,
            approach="MLP",
            video_names = [f"{skill_name}_trial_{i}" for i in range(number_of_saved_trials(skill_name))],
            latent_dim=8,
            framedrop=OnlyLabelledFramesDroppingPolicy,
            features=StampedLatentObservationsRiskLabels,
            train=True,
        )
    assert acc > 92

def test_E6():
    set_session("test_session")
    skill_name='peg_place'
    acc = evaluate_hyperparams(skill_name,
        approach="MLP",
        video_names = [f"{skill_name}_trial_{i}" for i in range(number_of_saved_trials(skill_name))],
        latent_dim=8,
        framedrop=OnlyLabelledFramesDroppingPolicy,
        features=StampedLatentObservationsRiskLabels,
    )
    assert acc > 85

def test_E7():
    set_session("test_session")
    skill_name='peg_door'
    acc = evaluate_hyperparams(skill_name+'_trial_0',
        approach="MLP",
        video_names = [f"{skill_name}_trial_{i}" for i in range(number_of_saved_trials(skill_name))],
        latent_dim=8,
        framedrop=OnlyLabelledFramesDroppingPolicy,
        features=StampedLatentObservationsRiskLabels,
    )
    assert acc > 90

def test_E8():
    set_session("test_session")
    skill_name='peg_pick'
    acc = evaluate_hyperparams(skill_name+'_trial_0',
        approach="MLP",
        video_names = [f"{skill_name}_trial_{i}" for i in range(number_of_saved_trials(skill_name))],
        latent_dim=8,
        framedrop=OnlyLabelledFramesDroppingPolicy,
        features=StampedLatentObservationsRiskLabels,
    )
    assert acc > 90

def test_E9():
    set_session("test_session")
    skill_name='peg_place'
    acc = evaluate_hyperparams(skill_name+'_trial_0',
        approach="MLP",
        video_names = [f"{skill_name}_trial_{i}" for i in range(number_of_saved_trials(skill_name))],
        latent_dim=8,
        framedrop=OnlyLabelledFramesDroppingPolicy,
        features=StampedLatentObservationsRiskLabels,
    )
    assert acc > 90

def test_E10():
    set_session("peg_door_session")
    skill_name = 'peg_door'
    acc = evaluate_hyperparams(skill_name,
        approach="MLP",
        video_names = all_trial_names(skill_name, include_repr=True),
        latent_dim=8,
        framedrop=OnlyLabelledFramesDroppingPolicy,
        features=StampedLatentObservationsRiskLabels,
    )
    assert acc > 96

def test_E11():
    set_session("peg_door_session")
    skill_name = 'peg_door'
    acc = evaluate_hyperparams(skill_name,
        approach="GP",
        video_names = all_trial_names(skill_name, include_repr=True),
        latent_dim=8,
        framedrop=OnlyLabelledFramesDroppingPolicy,
        features=StampedLatentObservationsRiskLabels,
        
    )
    assert acc > 94

def test_E12():
    set_session("peg_door_session")
    skill_name = 'peg_door'
    acc = evaluate_hyperparams(skill_name,
        approach="DistLS",
        video_names = all_trial_names(skill_name, include_repr=True),
        latent_dim=8,
        framedrop=OnlyLabelledFramesDroppingPolicy,
        features=StampedLatentObservationsRiskLabels,
        
    )
    assert acc > 80

def test_E13():
    set_session("peg_door_session")
    skill_name = 'peg_door'
    acc = evaluate_hyperparams(skill_name,
        approach="LR",
        video_names = all_trial_names(skill_name, include_repr=True),
        latent_dim=8,
        framedrop=OnlyLabelledFramesDroppingPolicy,
        features=StampedLatentObservationsRiskLabels,
        
    )
    assert acc > 70

def test_E14():
    set_session("peg_door_session")
    skill_name = 'peg_door'
    acc = evaluate_hyperparams(skill_name,
        approach="DistLR",
        video_names = all_trial_names(skill_name, include_repr=True),
        latent_dim=8,
        framedrop=OnlyLabelledFramesDroppingPolicy,
        features=StampedLatentObservationsRiskLabels,
    )
    if acc < 70:
        acc = evaluate_hyperparams(skill_name,
            approach="DistLR",
            video_names = all_trial_names(skill_name, include_repr=True),
            latent_dim=8,
            framedrop=OnlyLabelledFramesDroppingPolicy,
            features=StampedLatentObservationsRiskLabels,
            train = True,
        )
    assert acc > 70

def test_E15():
    set_session("peg_door_session")
    skill_name = 'peg_door'
    acc = evaluate_hyperparams(skill_name,
        approach="DistMin",
        video_names = all_trial_names(skill_name),
        latent_dim=8,
        framedrop=OnlyLabelledFramesDroppingPolicy,
        features=StampedLatentObservationsRiskLabels,
    )
    if acc < 73:
        acc = evaluate_hyperparams(skill_name,
            approach="DistMin",
            video_names = all_trial_names(skill_name),
            latent_dim=8,
            framedrop=OnlyLabelledFramesDroppingPolicy,
            features=StampedLatentObservationsRiskLabels,
            train = True,
        )

    assert acc > 73


def test_custom_DistMin_has_0FalseNegatives():
    set_session("peg_door_session")
    video_embedder = VideoEmbedder(8)
    video_embedder.load("peg_door")  # load data
    video_embedder.load_model(path=video_embedding.path + "/saved_models/")

    risk_estimator = get_risk_estimator("peg_door", "DistMin", StampedLatentObservationsRiskLabels, 8, video_embedder)
    
    train_dl, test_dl = RiskEstimationDataset.load(
        video_names=all_trial_names("peg_door", include_repr=True),
        video_embedder=video_embedder,
        batch_size=video_embedder.batch_size,
        frame_dropping_policy=OnlyLabelledFramesDroppingPolicy,
        features=StampedLatentObservationsRiskLabels,
    )

    
    risk_estimator.training_loop(
        train_dl, num_epochs=2500, patience=10000
    )  # Train Risk Aware module
    risk_estimator.save_model(
        path=risk_estimation.path + "/saved_models/"
    )

    X, Y = RiskEstimationDataset.dataloader_to_array(test_dl)
    Y = Y.cpu().numpy().squeeze()
    Y_pred, _ = risk_estimator.sample(X)

    tn, fp, fn, tp = confusion_matrix(Y, Y_pred).ravel()
    
    assert fn == 0

def test_custom_DistLR_geq_DistLS_geq_DistMin():
    ''' Accruacy of DistLR >= DistLS >= DistMin 
    Logistic regression (DistLR) precisely choose the threshold
    Linear search (DistLS) uses predefined Linspace, so accuracy must be equal or worse
    DistMin prioritizes safety, so it must be equal or worse than DistLS
    '''
    set_session("peg_door_session")
    skill_name = 'peg_door'
    acc_distls = evaluate_hyperparams(skill_name,
        approach="DistLS",
        video_names = all_trial_names(skill_name, include_repr=True),
        latent_dim=8,
        framedrop=OnlyLabelledFramesDroppingPolicy,
        features=StampedLatentObservationsRiskLabels,
        train=True,
    )
    acc_distlr = evaluate_hyperparams(skill_name,
        approach="DistLR",
        video_names = all_trial_names(skill_name, include_repr=True),
        latent_dim=8,
        framedrop=OnlyLabelledFramesDroppingPolicy,
        features=StampedLatentObservationsRiskLabels,
        train=True,
    )
    acc_distmin = evaluate_hyperparams(skill_name,
        approach="DistMin",
        video_names = all_trial_names(skill_name),
        latent_dim=8,
        framedrop=OnlyLabelledFramesDroppingPolicy,
        features=StampedLatentObservationsRiskLabels,
        train=True,
    )
    assert acc_distlr >= acc_distls >= acc_distmin

def custom_mlp_arch(arch='A1', skill_name = 'peg_door', video_latent_dim=8):
    set_session("arch_test_session")
    # Video embedder encodes skill from video
    video_embedder = VideoEmbedder(video_latent_dim)
    video_embedder.load(skill_name)  # load data

    # video_embedder.create_video(path=risk_estimation.path + "/videos/")  # train ae
    video_embedder.load_model(
        path=video_embedding.path + "/saved_models/"
    )  # OR load model

    risk_estimator = MLPRiskEstimator(name=skill_name, 
        xdim=StampedLatentObservationsRiskLabels.xdim(video_latent_dim),
        arch=arch)

    # Risk Estimator loads data for training
    train_dl, test_dl = RiskEstimationDataset.load(
        video_names=all_trial_names(skill_name, include_repr=True),
        video_embedder=video_embedder,
        batch_size=video_embedder.batch_size,
        frame_dropping_policy=OnlyLabelledFramesDroppingPolicy,
        features=StampedLatentObservationsRiskLabels,
    )

    risk_estimator.training_loop(
        train_dl, num_epochs=2500, patience=10000
    )  # Train Risk Aware module
    risk_estimator.save_model(
        path=risk_estimation.path + "/saved_models/"
    )

    X, Y = RiskEstimationDataset.dataloader_to_array(test_dl)
    
    Y = Y.cpu().numpy().squeeze()
    Y_pred, _ = risk_estimator.sample(X)

    acc = 100 * (Y == Y_pred).mean()
    print(acc)
    return acc

def test_custom_mlp_all_arch():
    for arch in ['A1', 'A2', 'A3', 'A4']:
        custom_mlp_arch(arch=arch)    


if __name__ == '__main__':
    # test_E4()
    # test_E5()
    # test_E6()
    # test_E7()
    # test_E8()
    # test_E9()
    test_E10()
    test_E11()
    test_E12()
    test_E13()
    test_E14()
    test_E15()