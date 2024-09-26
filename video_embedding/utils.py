import os
import numpy as np
import cv2
import pandas as pd
import skvideo.io
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import risk_estimation



def load(file='last'):
    import rospkg
    ros_pack = rospkg.RosPack()
    _package_path = ros_pack.get_path('trajectory_data')
    data = np.load(f"{_package_path}/trajectories/{get_session()}/{file}.npz")
    return data

def set_session(name):
    ''' Save to subdirectory '''
    global session
    session = name
    # print(f"[{__name__}] session is set ", name)

def get_session():
    global session
    try:
        session
    except NameError:
        session = ""
    # print(f"[{__name__}] session is read ", session)
    return session

def load_video(name):
    data = load(file=name)
    images= data['img']
    images_new=np.zeros((len(images),64,64))

    for i in range(len(images)):
        images_new[i]=cv2.resize(images[i], (64, 64))

    # images_new = images_new[:, np.newaxis, :, :]

    return images_new

def tensor_image_to_cv2(image):
    ''' Returns image ready for imshow '''
    image = image.detach().cpu().numpy().squeeze().astype(np.uint8)
    image = cv2.resize(image, (64, 64))
    return image
    
def visulize_video(img):
    import cv2
    # Assuming you have an array of images called 'images'
    for image in img:
        image = (image * 255).astype(np.uint8)
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video', 800, 600) 
        cv2.imshow('Video',image)
        if cv2.waitKey(25) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cv2.destroyAllWindows()

def save_video(path, name, tensor_images, h=64, w=64):
    try: # video images from gpu
        tensor_images = tensor_images.cpu().detach().numpy().squeeze()
    except AttributeError: # video images from cpu
        pass 
    # images_reconstruct= np.squeeze(output, axis=1)

    output_file = path + name + '.mp4'
    frame_rate = 30
    
    print(f"Saving video: {name} with {len(tensor_images)} frames and lenth {round((len(tensor_images)/30.), 1)}s.")
    writer = skvideo.io.FFmpegWriter(output_file, inputdict={'-framerate': str(frame_rate)}, outputdict={'-vcodec': 'libx264'})
    for i in range(len(tensor_images)):
        writer.writeFrame(tensor_images[i].reshape(h, w).astype(np.uint8))
    writer.close()

def visualize_labelled_video(images, labels={}, press_for_next_frame=False, printer=False, h=64, w=64):
    '''
    '''
    risk_flag = np.zeros((len(images)))
    safe_flag = np.zeros((len(images)))
    novelty_flag = np.zeros((len(images)))
    recovery_phase = -1.0 * np.ones((len(images)))

    if 'risk_flag' in labels:
        print("Risk flag")
        if labels['risk_flag'].squeeze().ndim != 0:
            risk_flag = labels['risk_flag'].squeeze()
    if 'safe_flag' in labels:
        print("Safe flag")
        if labels['safe_flag'].squeeze().ndim != 0:
            safe_flag = labels['safe_flag'].squeeze()
    if 'novelty_flag' in labels:
        print("Novelty flag")
        if labels['novelty_flag'].squeeze().ndim != 0:
            novelty_flag = labels['novelty_flag'].squeeze()
    if 'recovery_phase' in labels:
        print("recovery_phase")
        if labels['recovery_phase'].squeeze().ndim != 0:
            recovery_phase = labels['recovery_phase'].squeeze()

    for n, image in enumerate(images):
        if visualize_labelled_video_frame(
            image, 
            risk_flag[n],
            safe_flag[n],
            novelty_flag[n],
            recovery_phase[n],
            press_for_next_frame=press_for_next_frame,
            printer=printer,
            h=h, w=w):
            break
        

def visualize_labelled_video_frame(image, risk_flag, safe_flag=0, novelty_flag=0, recovery_phase=-1.0, press_for_next_frame=False, printer=False, w=64, h=64):
    if risk_flag:
        risk_label = 'RISK!'
        color=(0,0,255)
    elif safe_flag:
        risk_label = 'SAFE'
        color=(255,0,0)
    else:
        risk_label = 'SAFE'
        color=(255,0,0)
        risk_label = ''
    
    if novelty_flag:
        novelty_label = 'N'
    else:
        novelty_label = ''
    
    if recovery_phase != -1.0:
        recovery_phase = str(round(recovery_phase,1))
    else:
        recovery_phase = ""

    image = image.squeeze().astype(np.uint8)
    image = cv2.resize(image, (64, 64))

    cv2.putText(image, risk_label, (0, 12), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, color, 1, 2)
    cv2.putText(image, novelty_label, (0, 64-12), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (255, 0, 0), 1, 2)
    cv2.putText(image, recovery_phase, (40, 64-12), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (255, 0, 0), 1, 2)

    
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    # cv2.moveWindow("Image",1200,30)
    cv2.resizeWindow("Image", 640, 640)
    cv2.imshow("Image", image)

    if printer: print(f"R: {risk_flag}, S: {safe_flag}")
    if press_for_next_frame:
        cv2.waitKey(0)  # Wait for a key press to close the window
    if cv2.waitKey(25) & 0xFF == 27:  # Press 'Esc' to exit
        return True
    return False


def visualize_labelled_video_frame_inline(image, risk_flag, safe_flag=0, novelty_flag=0, recovery_phase=-1.0,press_for_next_frame=False, printer=False):
    if risk_flag:
        risk_label = 'R'
    elif safe_flag:
        risk_label = 'S'
    else:
        risk_label = ''
    
    if novelty_flag:
        novelty_label = 'N'
    else:
        novelty_label = ''

    if recovery_phase != -1.0:
        recovery_phase = str(round(recovery_phase, 1))
    else:
        recovery_phase = ""

    # Assume 'image' is a numpy array loaded in your environment
    # and 'risk_label', 'novelty_label', 'risk_flag', 'safe_flag' are defined
    image = image.squeeze().astype(np.uint8)
    image = cv2.resize(image, (64, 64))

    # Convert the color from BGR to RGB (matplotlib expects RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Using cv2 to put text on image
    cv2.putText(image, risk_label, (0, 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, novelty_label, (0, 64-12), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, recovery_phase, (40, 64-12), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 0), 1, cv2.LINE_AA)
    

    # Now using matplotlib to display the image
    plt.figure(figsize=(2, 2))  # Size is adjustable
    plt.imshow(image)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()


def number_of_saved_trials(video: str):
    import rospkg
    ros_pack = rospkg.RosPack()
    _package_path = ros_pack.get_path('trajectory_data')

    n = 0
    while os.path.isfile(f'{_package_path}/trajectories/{get_session()}/{video}_trial_{n}.npz'):
        n += 1
    
    return n

def all_trial_names(skills: str, include_repr: bool = True):
    if isinstance(skills, str): # skills is single skill
        skills = [skills]

    names = []
    for skill in skills:
        
        names.extend([f"{skill}_trial_{i}" for i in range(number_of_saved_trials(skill))])
        if include_repr:
            names.append(skill)
    
    return names

def number_of_saved_test_trials(video: str):
    import rospkg
    ros_pack = rospkg.RosPack()
    _package_path = ros_pack.get_path('trajectory_data')

    n = 0
    while os.path.isfile(f'{_package_path}/trajectories/{get_session()}/{video}_test_{n}.npz'):
        n += 1
    
    return n

def all_test_names(skills: str, include_repr: bool = False):
    if isinstance(skills, str): # skills is single skill
        skills = [skills]

    names = []
    for skill in skills:
        
        names.extend([f"{skill}_test_{i}" for i in range(number_of_saved_test_trials(skill))])
        if include_repr:
            names.append(skill)
    
    return names


def load_latent_trajectory(name, latent_dim, path=None):
    file_path = path + name + '_latent_' + str(latent_dim) + '.npz'
    data = np.load(file_path)
    latent_traj = data['latent_traj']
    return latent_traj
   

def behaviour_trial_names(skills: str, behaviours=[], include_repr=True):
    if isinstance(skills, str): # skills is single skill
        skills = [skills]
    
    import rospkg
    ros_pack = rospkg.RosPack()
    _package_path = ros_pack.get_path('trajectory_data')

    video_names = []    
    for skill in skills:
        df = pd.read_csv(f'{_package_path}/trajectories/{get_session()}/{skill}_description.csv')
        dfdict = df.to_dict()
        for video_n in range(len(dfdict['n'])):
            if dfdict['Behaviours'][video_n] in behaviours:
                video_names.append(dfdict['Videos'][video_n])

    return video_names


def visualize_data_sne(X, Y):
    tsne = TSNE(n_components=2, verbose=1, perplexity=5, n_iter=400)
    tsne_results = tsne.fit_transform(X)

    plt.figure(figsize=(12, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=Y, alpha=0.5, cmap='viridis')
    plt.title(f't-SNE Visualization of High-Dimensional Data {tsne_results.shape}')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.show()

def trajectory_data_shape_printer(data):
    for k in data.keys():
        print(f"{k}: {data[k].shape}")

def clip_samples(data, limit=400):
    print("Samples before: ")
    trajectory_data_shape_printer(data)
    for k in data.keys():
        if k == 'img':
            data[k] = data[k][0:limit]
        else:
            data[k] = data[k][:,0:limit]
    print("Samples after: ")
    trajectory_data_shape_printer(data)
    return data

def put_riskdist_on_image(image, risk):
    cv2.putText( # Write on the image prob. of risk
        image, str(risk.round(1)), (0, 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (255 * int(image[0:20, 0:20].mean() < (255 / 2)), 0, 0),
        1, 2,
    )
    return image

def put_risklabel_on_image(image, label):
    cv2.putText( # Write on the image R for risk, S for safe
        image, label, (0, 61),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (255 * int(image[44:, 0:20].mean() < (255 / 2)), 0, 0),
        1, 2,
    )
    return image

def save_models_index_list(path: str, video_name):
    result_names = list_files_in_folder(path, ext=".csv", cut_substr="/")
    result_names = [r[:-4] for r in result_names] # discard .csv extension
    result_names = [r.split(video_name+"_")[-1] for r in result_names] # discard video_name extension
    result_names = sorted(result_names)
    for n,result_name in enumerate(result_names): # remove result_index_list file from list
        if "result_index_list" in result_name:
            result_names.pop(n)
    
    df = pd.DataFrame(np.array([result_names]).T, columns=['Results'])
    df.to_csv(f"{path}/{video_name}_result_index_list.csv", index_label='Time')


def save_video_index_list(path: str, parent_folder="videos"):
    video_names = list_files_in_folder(path, ext=".mp4", cut_substr=f"/{parent_folder}/")
    video_names = [video[:-4] for video in video_names] # discard .mp4 extension

    video_names = sorted(video_names)

    df = pd.DataFrame(np.array([video_names]).T, columns=['Videos'])
    df.to_csv(f"{risk_estimation.path}/{parent_folder}/video_list.csv", index_label='Time')

def list_files_in_folder(path, ext=".mp4", cut_substr="/videos/"):
    # all_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    # with searching subfolders
    all_files = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(path) for f in filenames]
    files_with_ext = []
    for file in all_files:
        if file[-len(ext):] == ext:
            files_with_ext.append(file.split(cut_substr)[-1])
    return files_with_ext