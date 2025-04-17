"""Making simulated dataset on labelled one + training
"""
import argparse
from risk_estimation.models.safety_layer import get_risk_estimator
from risk_estimation.datasets.frame_dropping import *
from risk_estimation.datasets.risk_dataloader import RiskEstimationDataset as D
from risk_estimation.datasets.risk_feature_extractor import *
from video_embedding.models.video_embedder import VideoEmbedder
from video_embedding.utils import all_trial_names, set_session, all_test_names
from risk_estimation.result_evaluator import benchmark_eval

import matplotlib.pyplot as plt
import numpy as np



def main(args):
    if args.session != "":
        set_session(args.session)

    video_embedder = VideoEmbedder(name=args.skill_name, nn_model="Autoencoder3")
    video_embedder.load_model()

    features = eval("StampedLatentObservationsRiskLabels")
    framedrop_policy = eval("OnlyLabelledFramesDroppingPolicy")

    risk_estimator = get_risk_estimator(
        'TwinGP', args.skill_name, features.xdim(video_embedder.latent_dim), video_embedder, out_assessment="cautious", train_patience=500, train_epoch=500
    )

    test_dataloader = D.load_dataloader(all_test_names(args.skill_name), video_embedder, 64, framedrop_policy, features)
    novel_dataloader = D.load_dataloader(all_test_names(args.skill_name), video_embedder, 64, framedrop_policy.novel(), features.novel())
    
    risk_estimator.set_dataloaders_for_validation([
        test_dataloader, 
        novel_dataloader
    ], names=["test", "novel"])

    saved_logs = {"n_executions": [], "acc_test": [], "tn_test": [], "fp_test": [], "fn_test": [], "tp_test": [], "acc_novel": [], "tn_novel": [], "fp_novel": [], "fn_novel": [], "tp_novel": []}
    for n in range(1,len(all_trial_names(args.skill_name))):
        train_video_names = all_trial_names(args.skill_name)[:n]
        print(f"[{n}] Train with trials: {train_video_names}")

        dataloader = D.load_dataloader(train_video_names, video_embedder, 64, framedrop_policy, features)
        risk_estimator.training_loop(dataloader)

        benchmark_eval("Train_dataset", args.skill_name, dataloader.dataset, video_embedder, risk_estimator)
        e_test = benchmark_eval("Test_dataset", args.skill_name, test_dataloader.dataset, video_embedder, risk_estimator)
        e_novel = benchmark_eval("Novel_Dataset", args.skill_name, novel_dataloader.dataset, video_embedder, risk_estimator)
        saved_logs["n_executions"].append(n)
        saved_logs["acc_test"].append(e_test.acc_score)
        saved_logs["tn_test"].append(e_test.tn)
        saved_logs["fp_test"].append(e_test.fp)
        saved_logs["fn_test"].append(e_test.fn)
        saved_logs["tp_test"].append(e_test.tp)
        saved_logs["acc_novel"].append(e_novel.acc_score)
        saved_logs["tn_novel"].append(e_novel.tn)
        saved_logs["fp_novel"].append(e_novel.fp)
        saved_logs["fn_novel"].append(e_novel.fn)
        saved_logs["tp_novel"].append(e_novel.tp)
    real_exp_plot(saved_logs)

def real_exp_plot(d):
    # --- Plot 1: Accuracy Curves ---
    plt.figure()
    plt.plot(d['n_executions'], d['acc_test'], marker='o', label='Test Accuracy')
    plt.plot(d['n_executions'], d['acc_novel'], marker='o', label='Novel Accuracy')
    plt.xlabel('Number of Executions in Training')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of Training Executions')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_vs_executions.pdf")
    plt.show()
    

    # --- Plot 2: Stacked Bar of Confusion Matrix Components ---
    width = 0.35
    x = np.arange(len(d['n_executions']))

    # Plot 2: Improved Stacked Bar Chart with Combined Legend
    plt.figure(figsize=(5, 3))
    # Plotting Test Dataset Bars
    bar_test_tn = plt.bar(x - width/2, d['tn_test'], width)
    bar_test_fp = plt.bar(x - width/2, d['fp_test'], width, bottom=d['tn_test'])
    bar_test_fn = plt.bar(x - width/2, d['fn_test'], width, bottom=np.array(d['tn_test'])+np.array(d['fp_test']))
    bar_test_tp = plt.bar(x - width/2, d['tp_test'], width, bottom=np.array(d['tn_test'])+np.array(d['fp_test'])+np.array(d['fn_test']))

    # Plotting Novel Dataset Bars
    plt.bar(x + width/2, d['tn_novel'], width)
    plt.bar(x + width/2, d['fp_novel'], width, bottom=d['tn_novel'])
    plt.bar(x + width/2, d['fn_novel'], width, bottom=np.array(d['tn_novel'])+np.array(d['fp_novel']))
    plt.bar(x + width/2, d['tp_novel'], width, bottom=np.array(d['tn_novel'])+np.array(d['fp_novel'])+np.array(d['fn_novel']))

    # Creating unified legend for the components only
    legend_labels = ['TN', 'FP', 'FN', 'TP']
    legend_handles = [
        bar_test_tn[0],
        bar_test_fp[0],
        bar_test_fn[0],
        bar_test_tp[0]
    ]

    plt.xlabel('Number of Executions in Training')
    plt.ylabel('Count')
    plt.title('Confusion Matrix Components by Dataset')
    plt.xticks(x, d['n_executions'])
    plt.legend(legend_handles, legend_labels, title='Components', loc='lower left')
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig("confusion_matrix_components.pdf")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Custom Train and Evaluate Risk Estimator",
        description="",
        epilog="",
    )
    parser.add_argument("-n", "--skill_name", default="peg_pick404")
    parser.add_argument("-s", "--session", default="AE3")

    main(parser.parse_args())