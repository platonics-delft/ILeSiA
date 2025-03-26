import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import os
import cv2
from pathlib import Path
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# Put here the autogen (results forlder) for which you want to generate plots
# e.g.
root_dir = "/home/petr/Downloads/ilesia_25_03/autogen"
frame_dropping = True

def skill_framedropping(data, filepath):
    if "peg_pick404" in filepath:
        for i in range(len(data["RiskTrue"])):
            if not ((60 < i < 90) or (480 < i < 510)):
                data["RiskTrue"][i] = 0.0
                data['SafeTrue'][i] = 0.0
    elif "peg_door404" in filepath:
        for i in range(len(data["RiskTrue"])):
            if not ((150 < i < 240) or (630 < i < 710)):
                data["RiskTrue"][i] = 0.0
                data['SafeTrue'][i] = 0.0 
    elif "peg_place404" in filepath:
        for i in range(len(data["RiskTrue"])):
            if not ((60 < i < 150) or (400 < i < 460)):
                data["RiskTrue"][i] = 0.0
                data['SafeTrue'][i] = 0.0 
    return data

def plotter(filepath, half = "", connection_line=True, series_enabled=True):
    print(filepath)

    if "GP+L" in filepath:
        name = "Linear + $\mathcal{GP}$"
    elif "GP" in filepath:
        name = "$\mathcal{GP}$"
    elif "TwinGP" in filepath:
        name = "Twin $\mathcal{GP}$"
    elif "MLP" in filepath:
        name = "$\mathcal{MLP}$"

    plt.rcParams["pdf.use14corefonts"] = True
    plt.rcParams["ps.useafm"] = True

    data = pd.read_csv(filepath)
    if frame_dropping:
        data = skill_framedropping(data, filepath)

    if half != "":
        h = (data.index[-1] // 2)
        if half == "left":
            # Frames numbers that are plotted in the figure
            # Right now tuned for two frames
            frame_numbers = np.array(np.linspace(0, h, 4), dtype=int)[1:3]
            data = data.iloc[:h]
        elif half == "right":
            frame_numbers = np.array(np.linspace(h, data.index[-1], 4), dtype=int)[1:3]
            data = data.iloc[h:]
        else: raise Exception()
    else:
        frame_numbers = np.array(np.linspace(0, data.index[-1], 6), dtype=int)
    data['Risk'] = data['Risk'].astype(float)
    data['Correct'] = data['Correct'].astype(int)

    cropped_frame = extract_image(filepath, frame_numbers=frame_numbers)
    if cropped_frame is None:
        return 
    # Create figure and axes using the subplots function
    if half == "":
        fig, axs = plt.subplots(2, 1, figsize=(6, 4))
    else:
        fig = plt.figure(figsize=(3, 4))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])  # Top plot 3 times taller than bottom
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        axs = [ax1, ax2]

    ax = axs[1]
    # Display the image in the top subplot
    axs[0].imshow(cropped_frame)
    # Remove x and y ticks
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    text_offset = 0.15  # Adjust as needed based on your figure size and DPI
    if half != "": # is left or right
        fig.text(0.22, 0.82, 'Original', rotation='vertical', va='center', fontsize=8)
        fig.text(0.22, 0.57, 'Reconstructed', rotation='vertical', va='center', fontsize=8)
        fig.text(0.16, 0.44, 'Loss', rotation='horizontal', va='center', fontsize=8)
    else: # full
        fig.text(0.2, 0.83, 'Original', rotation='vertical', va='center', fontsize=8)
        fig.text(0.2, 0.67, 'Reconstructed', rotation='vertical', va='center', fontsize=8)
        fig.text(0.18, 0.555, 'Loss', rotation='horizontal', va='center', fontsize=8)

    for n,frame_n in enumerate(frame_numbers):
        # Annotate a line connecting the bottom x-axis point to the image
        axs[1].axvline(x=frame_n, color='black', linestyle='--', zorder=12, linewidth=1.0, ymin=0.06, ymax=0.94)
        
        if connection_line:
            xy = (frame_n, 1)  # Endpoint in data coordinates for the bottom plot
            xy2 = (n/len(frame_numbers)+1/(len(frame_numbers)*2), 0)  # Start point in axes fraction for the top plot
            con = patches.ConnectionPatch(xyA=xy2, xyB=xy, coordsA='axes fraction', coordsB='data', 
                                    axesA=axs[0], axesB=axs[1], color="black")
            fig.add_artist(con)

    # Plot vertical lines based on condition
    # for i in range(data.index[0], data.index[-1]):
    #     if data['RiskTrue'][i] == 1 or data['SafeTrue'][i] == 1:
    #         color = 'blue' if data['Correct'][i] == 1 else 'red'
    #         ax.axvline(x=i, color=color, alpha=0.1, zorder=2)
    
    if series_enabled:
        SLIDINGWINDOW = 5
        for i in range(data.index[0], data.index[-1]):
            if i < SLIDINGWINDOW: continue
            
            if (np.array(data['Risk'][i-5:i]) > 0.51).all():
                ax.plot(np.clip(data['Risk'][i-5:i], 0, 1), color="red", linewidth=1.5, zorder=12)
                # ax.axvline(x=i, color="red", ymin=(0.06/1.12), ymax=0.06+(data['Risk'][i]/1.12), zorder=8)

    # Plot the risk data
    if series_enabled:
        series_enabled_text = ""
        ax.plot(np.clip(data['Risk'] - data['Std'], 0, 1), label='$\mu$', color="green", linewidth=1, zorder=10)
        ax.plot(np.clip(data['Risk'], 0, 1), label='$r<0.5$ (safe)', color="blue", linewidth=1, zorder=10)
        
        ax.fill_between(data.index, 
                np.clip(data['Risk'], 0, 1), 
                np.clip(data['Risk'] - 2*data['Std'], 0, 1), 
                color='green', alpha=0.3, label='$\mu$ ± $\sigma$', zorder=9, hatch="///")
    else:
        series_enabled_text = "_nodata"
    # Add a dashed horizontal line
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1, zorder=6)
    ax.axhline(y=1.0, color='black', linestyle='-', linewidth=1, zorder=6)
    ax.axhline(y=0.0, color='black', linestyle='-', linewidth=1, zorder=6)

    # Add vertical text for "Safe" below the line and "Risk" above the line
    ax.text(data.index[-1]+5, 0.25, 'Safe', rotation='vertical', verticalalignment='center', fontsize=12, zorder=12)
    ax.text(data.index[-1]+5, 0.75, 'Risk', rotation='vertical', verticalalignment='center', fontsize=12, zorder=12)
    ax.text(data.index[-1]+5, 0.5, '$\\tau$', verticalalignment='center', fontsize=12, zorder=12)

    if half != "":
        fig.text(0.0, 0.23, name, rotation='vertical', va='center', fontsize=15)
    else: # full
        fig.text(0.1, 0.3, name, rotation='vertical', va='center', fontsize=15)

    # Plot markers for RiskFlag and SafeFlag
    YD = 1.03
    ax.scatter(data[data['RiskTrue'] == 1].index, YD*np.ones(data[data['RiskTrue'] == 1]['Risk'].shape), color='red', marker="*", zorder=10)
    if len(data[data['RiskTrue'] == 1].index) > 0:
        prev = False
        for i in data[data['RiskTrue'] == 1].index:
            if prev and (i - prev) > 5:
                ax.scatter(i, YD, color='black', marker="*", zorder=12)
                ax.scatter(prev, YD, color='black', marker="*", zorder=12)
            prev = i
        ax.scatter(data[data['RiskTrue'] == 1].index[0], YD, color='black', marker="*", zorder=12)
        ax.scatter(data[data['RiskTrue'] == 1].index[-1], YD, color='black', marker="*", zorder=12)

    ax.scatter(data[data['SafeTrue'] == 1].index, (1-YD)*np.ones(data[data['SafeTrue'] == 1]['Risk'].shape), color='green', marker="*", zorder=12)
    if len(data[data['SafeTrue'] == 1].index) > 0:
        prev = False
        for i in data[data['SafeTrue'] == 1].index:
            if prev and (i - prev) > 5:
                ax.scatter(i, 1-YD, color='black', marker="*", zorder=12)
                ax.scatter(prev, 1-YD, color='black', marker="*", zorder=12)
            prev = i
        ax.scatter(data[data['SafeTrue'] == 1].index[0], (1-YD), color='black', marker="*", zorder=12)
        ax.scatter(data[data['SafeTrue'] == 1].index[-1], (1-YD), color='black', marker="*", zorder=12)

    # Set labels
    ax.set_xlabel('$\\alpha$')
    ax.set_ylabel('r', rotation='horizontal')

    # Set the x-axis limits
    ax.set_xlim(left=data.index[0]-5, right=data.index[-1])
    ax.set_ylim(bottom=-0.06, top=1.06)

    # Use tight_layout to adjust the layout
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
    # fig.tight_layout()


    handles, labels = ax.get_legend_handles_labels()
    import matplotlib.lines as mlines
    small_patch = mlines.Line2D([], [], color='red', linewidth=3, label='$r>0.5$ (risk!)')

    handles.insert(0, small_patch)
    labels.insert(0, '$r>0.5$ (risk!)')
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple
    small_patch = mpatches.Patch(color="red", label='Ground Truth', linewidth=1.0)
    small_patch2 = mpatches.Patch(color="green", label='Ground Truth', linewidth=1.0)
    star_marker = Line2D([0], [0], color="black", marker="*", markersize=5, linestyle="None")

    handles.extend([(star_marker, small_patch, small_patch2)])
    labels.extend(['Ground Truth'])


    legend = ax.legend(handles, labels, fontsize="xx-small", loc='lower left', handler_map={tuple: HandlerTuple(ndivide=None)})
    legend.set_zorder(15)


    # plt.show()
    if connection_line:
        fig.savefig(f'{filepath[:-4]}_plot{half}{series_enabled_text}.pdf', dpi=300, format="pdf")
        fig.savefig(f'{filepath[:-4]}_plot{half}{series_enabled_text}.png', dpi=300, format="png")
    else:
        fig.savefig(f'{filepath[:-4]}_plot{half}{series_enabled_text}__.pdf', dpi=300, format="pdf")
        fig.savefig(f'{filepath[:-4]}_plot{half}{series_enabled_text}__.png', dpi=300, format="png")

    # Close the figure
    plt.close(fig)



def extract_image(filepath, frame_numbers):
    folderpath = Path(filepath).parent
    f = []
    for (dirpath, dirnames, filenames) in os.walk(folderpath):
        f.extend(filenames)
        break
    videos_in_foldera = []
    for f_ in f:
        if f_[-4:] == ".mp4":
            videos_in_foldera.append(f_) 
    assert len(videos_in_foldera) <= 1
    if len(videos_in_foldera) == 0:
        return None
    video_filepath = videos_in_foldera[0]

    # Open the video file
    cap = cv2.VideoCapture(f"{folderpath}/{video_filepath}")

    frames = []
    for frame_number in frame_numbers:
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        ret, frame = cap.read()

        # Check if the frame has been correctly captured
        if ret:
            # Convert the image from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame[:144, :] # original + reconstructed images
            # frame = frame[:64, :] # only original image
            # frame = frame[64:128, :] # only reconstructed image 
        else:
            print(f"Failed to retrieve frame at index {frame_number}")

        frames.append(frame)
    frames = cv2.hconcat(frames)

    # Release the video capture object
    cap.release()

    return frames


for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        # Check if the file is a CSV and does not include "index" in the name
        # and does not start with "Test" or "Train"
        if file.endswith('.csv') and 'index' not in file and 'video_list' not in file and not (file.startswith('Test') or file.startswith('Train')):
            # Construct the full file path
            file_path = os.path.join(subdir, file)
            
            # plotter(file_path)
            # plotter(file_path, half="left")
            # plotter(file_path, half="right")
            plotter(file_path, connection_line=False, series_enabled=True)
            plotter(file_path, connection_line=True, series_enabled=True)
            # plotter(file_path, connection_line=False, series_enabled=False)
            # plotter(file_path, half="left", connection_line=False)
            # plotter(file_path, half="right", connection_line=False)
            

