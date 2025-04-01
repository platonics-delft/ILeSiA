
from video_embedding.utils import set_session
from test_final_benchmark import test_final_benchmarks

skills =[
    # "peg_pick404", 
    "peg_door404", 
    # "slider_move404", 
    # "slider_move404_2", # has better alignment between train and test trajectory data
    # "peg_place404", 
    # "probe_pick404"
    # "move_around404"
]
session = "quantitative_study"
stages = [
    [64,    "CustomResnetStage1"],  # stage 1, # needs stage-1 latent dim
    [256,   "CustomResnetStage2"],  # stage 2, # needs stage-2 latent dim
    [512,   "CustomResnetStage3"],  # stage 3, # needs stage-3 latent dim
    [1024,  "CustomResnetStage4"], # stage 4, # needs stage-4 latent dim
    [2048,  "CustomResnetStage5"], # stage 5, # needs stage-5 latent dim
]

set_session(session)
for skill_name in skills:
    save_video_flag = True
    for video_latent_dim,embedding_approach in stages:
        if isinstance(features, str):
            features = eval(features)

        print("save_video_flag ", save_video_flag)
        test_final_benchmarks(
            skill_name = skill_name,
            video_latent_dim = video_latent_dim,
            approach = 'resnet50',
            embedding_approach = embedding_approach,
            features = features,
            framedrop_policy = framedrop,
            out_assessment = out_assessment,
            train_epoch = 6000,
            train_patience = 6000,
            save_video_flag = save_video_flag,
        )
        save_video_flag = False