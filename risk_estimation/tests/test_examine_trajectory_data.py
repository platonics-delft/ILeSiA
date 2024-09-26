from video_embedding.utils import clip_samples, set_session, visulize_video, load
import argparse


def test_examine_trajectory_data():
    set_session("test_session")
    main(args={'file': 'peg_door_trial_0'})

def main(args):
    data = dict(load(file=args['file']))
    for k in data.keys():
        print(f"{k}: {data[k].shape}")
        print(data[k])
    # print(len(images))
    # visulize_video(images)
    clip_samples(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Play video",
        description="",
        epilog="",
    )
    parser.add_argument(
        "--file",
        default="peg_door_trial_0",
    )
    main(vars(parser.parse_args()))
