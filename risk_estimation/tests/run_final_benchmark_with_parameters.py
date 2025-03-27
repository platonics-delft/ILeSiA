from test_final_benchmark import test_final_benchmarks
from video_embedding.utils import all_test_names, all_trial_names, get_session, set_session
from risk_estimation.models.risk_estimation.frame_dropping import *

mapping = {
    "peg_pick404": "PegPick", 
    "peg_door404": "PegDoor", 
    "slider_move404": "SliderMove", 
    "slider_move404_2": "SliderMove", # has better alignment between train and test trajectory data
    "peg_place404": "PegPlace", 
    "probe_pick404": "ProbePick",
    "move_around404": "MoveAround",
}

skills = [
    # "peg_pick404", 
    "peg_door404", 
    # "peg_place404", 
    # "slider_move404", 
    # "slider_move404_2", # has better alignment between train and test trajectory data
    # "probe_pick404"
    # "move_around404"
] 
latent_dims = [
    12
]
approaches = [
    # 'LR',
    # 'MLP',
    # 'GP',
    'TwinGP',
    # 'resnet50',
]
filter_samples_to_label_areas = False

for latent_dim in latent_dims:
    for skill_name in skills:
        save_video_flag = True
        for approach in approaches:
            set_session("quantitative_study")
            if filter_samples_to_label_areas:
                framedrop_policy = eval(f"OnlyLabelledFramesDroppingPolicyRisk{mapping[skill_name]}")
            else:
                framedrop_policy = f"OnlyLabelledFramesDroppingPolicy"
            test_final_benchmarks(
                skill_name=skill_name,
                video_latent_dim=latent_dim,
                approach=approach,
                embedding_approach="LargeAutoencoder",
                features="StampedLatentObservationsRiskLabels",
                framedrop_policy=framedrop_policy,
                out_assessment="cautious",
                train_epoch=1500,
                train_patience=2000,
                save_video_flag=save_video_flag,
            )
            save_video_flag=False
