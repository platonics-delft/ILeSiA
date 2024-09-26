from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import cosine

from risk_estimation.models.risk_estimator import DistanceRiskEstimator, NMDistanceRiskEstimatorDTW
from video_embedding.utils import set_session
from video_embedding.models.video_embedder import RiskyBehavioralVideoEmbedder
import argparse
import cv2


def main(args):
    set_session(args['session'])
    video_embedder = RiskyBehavioralVideoEmbedder(name=args['video'], latent_dim=16)
    video_embedder.load(args['video'])

    encoded_traj = video_embedder.model.encoder(video_embedder.tensor_images).detach().cpu().numpy()
    tensor_images1 = deepcopy(video_embedder.tensor_images.detach().cpu().numpy())

    # images ~300 x latent space size 8
    video_embedder.name = args['video_test']
    video_embedder.load(args['video_test'])
    images2 = video_embedder.tensor_images
    encoded_traj_ood = video_embedder.model.encoder(images2).detach().cpu().numpy()

    tensor_images2 = deepcopy(video_embedder.tensor_images.detach().cpu().numpy())

    dre = DistanceRiskEstimator(args['video'], dist_fun=cosine, thr=0.5)
    dre.compare_trajectories(encoded_traj, encoded_traj)

    # %% Distance using Dynamic Time Warping

    distance, path = fastdtw(encoded_traj, encoded_traj, dist=cosine)
    path = np.array(path)

    correlation = dre.cross_test(encoded_traj, encoded_traj)
    # Plot the correlation matrix as a heatmap
    plt.figure()
    plt.title("Self Correlation map")
    heatmap = plt.imshow(correlation, cmap="hot", interpolation="nearest")
    plt.colorbar(heatmap)
    plt.plot(
        path[:, 1], path[:, 0], marker=".", color="b", markersize=1
    )  # Your original list
    print(f"DTW distance {distance}")


    distance, path = fastdtw(encoded_traj, encoded_traj_ood, dist=cosine)
    path = np.array(path)
    print(f"distance: {distance}")

    cross_correlation = dre.cross_test(encoded_traj, encoded_traj_ood)
    # Plot the correlation matrix as a heatmap
    plt.figure()
    plt.title("Cross Correlation map")
    heatmap = plt.imshow(cross_correlation, cmap="hot", interpolation="nearest")
    plt.colorbar(heatmap)
    plt.plot(
        path[:, 1], path[:, 0], marker=".", color="b", markersize=1
    )  # Your original list
    print(f"DTW distance {distance}")
    plt.show()


    # %%
    # Plot pictures with low dist
    for x, y in path:
        
        dist = cross_correlation[x, y]
        image1 = tensor_images1[x].astype(np.uint8)
        image1 = cv2.resize(image1, (64, 64))[0]
        image2 = tensor_images2[y].astype(np.uint8)
        image2 = cv2.resize(image2, (64, 64))[0]

        concatenated_image = np.hstack((image1, image2))

        cv2.putText(
            concatenated_image,
            str(dist.round(1)),
            (0, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            2,
        )

        cv2.namedWindow("Concatenated Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Concatenated Image", 1000, 500)
        cv2.imshow("Concatenated Image", concatenated_image)
        # if cv2.waitKey(25) & 0xFF == 27:  # Press 'Esc' to exit
        #     break
        cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Compare two demonstrations",
        description="",
        epilog="",
    )
    parser.add_argument(
        "--video",
        default="peg_door_trial_0",
    )
    parser.add_argument(
        "--video_test",
        default="peg_door_trial_1",
    )
    parser.add_argument(
        "--session",
        default="",
    )
    
    main(vars(parser.parse_args()))
