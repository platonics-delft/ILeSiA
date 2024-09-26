from test_final_benchmark import test_final_benchmarks
from video_embedding.utils import all_test_names, all_trial_names, get_session, set_session

mapping = {
    "peg_pick404": "PegPick", 
    "peg_door404": "PegDoor", 
    "slider_move404": "SliderMove", 
    "slider_move404_2": "SliderMove", # has better alignment between train and test trajectory data
    "peg_place404": "PegPlace", 
    "probe_pick404": "ProbePick",
    "move_around404": "MoveAround",
}

class Args():
    def __init__(self, skill, session, framedrop, learning_rate, epoch, patience, approach, out_assessment, features, embedding_approach, latent_dim):
        self.skill = skill
        self.session = session
        self.framedrop = framedrop
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.patience = patience
        self.approach = approach
        self.out_assessment = out_assessment
        self.features = features
        self.embedding_approach = embedding_approach
        self.latent_dim = latent_dim

skills = [
    # "peg_pick404", 
    "peg_door404", 
    # "slider_move404", 
    # "slider_move404_2", # has better alignment between train and test trajectory data
    # "peg_place404", 
    # "probe_pick404"
    # "move_around404"
] 
risks = [
    1,
    # 2
]
latent_dims = [
    12,
    # 16,
    # 24,
    # 32,
    # 48
]
approaches = [
    # 'LR',
    # 'MLP',
    'GP',
    # "L+GP",
    # "L+GP+1SKIP",
    # "L+GP+2SKIP",
    # 'resnet50',
]
out_assessments = [
    # "optimilstic",
    "cautious",
    # "reconstruction",
]

for latent_dim in latent_dims:
    for skill_name in skills:
        save_video_flag = True
        for out_assessment in out_assessments:
            for risk in risks:
                for approach in approaches:
                    if latent_dim == 12:
                        session = "manipulation_demo404_augment_12_session"
                    elif latent_dim == 16:
                        session = "manipulation_demo404_augment_16_session"
                    elif latent_dim == 24:
                        session = "manipulation_demo404_augment_24_session"
                    elif latent_dim == 32:
                        session = "manipulation_demo404_augment_32_session"
                    elif latent_dim == 48:
                        session = "manipulation_demo404_augment_48_session"
                    set_session(session)
                    framedrop_policy = f"OnlyLabelledFramesDroppingPolicy" #Risk{mapping[skill_name]}{risk}"

                    learning_rate = 0.01
                    train_epoch = 6000
                    train_patience = 2000
                    if latent_dim <= 16:
                        embedding_approach = "Autoencoder"
                    else:
                        embedding_approach = "LargeAutoencoder"
                    
                    features = "StampedDistRecErrLatentObservationsRiskLabels"
                    features = "StampedDistLatentObservationsRiskLabels"

                    test_final_benchmarks(
                        skill_name=skill_name,
                        video_latent_dim=latent_dim,
                        approach=approach,
                        embedding_approach=embedding_approach,
                        features=features,
                        framedrop_policy=framedrop_policy,
                        out_assessment=out_assessment,
                        train_epoch=train_epoch,
                        train_patience=train_patience,
                        save_video_flag=save_video_flag,
                    )
                    save_video_flag=False
