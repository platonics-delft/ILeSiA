
from copy import deepcopy
import numpy as np
try:
    import roboticstoolbox as rtb
except ModuleNotFoundError:
    rtb = None
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D

def get_panda_at_config(q):
    assert rtb is not None, "pip install roboticstoolbox-python"
    robot = rtb.models.Panda()
    
    robot.plot(q, backend='pyplot')
    # workaround to retrieve img
    path = "/tmp/tmp.png"
    plt.axis('off')
    plt.grid(b=None)
    plt.savefig(path)
    plt.close()
    return cv2.imread(path)

def plot_camera_images_along_robot_configurations(camera_images, robot_states, name="", images=5, single_image_size = 512):

    assert len(camera_images) == len(robot_states)
    
    idxs = np.array(np.linspace(0, len(camera_images)-1, images), dtype=int)
    camera_images = deepcopy(camera_images.astype(np.uint8).squeeze())

    concatenated_image = np.array([], dtype=np.uint8).reshape(2*single_image_size,0,3)
    for idx in idxs:
        camera_image = camera_images[idx]
        camera_image = cv2.cvtColor(camera_image, cv2.COLOR_GRAY2BGR)
        camera_image = cv2.resize(camera_image, (single_image_size,single_image_size), interpolation=cv2.INTER_AREA)

        panda_config = robot_states[idx]
        pandaimg = get_panda_at_config(q=panda_config)
        # pandaimg = cv2.cvtColor(pandaimg, cv2.COLOR_BGR2GRAY)
        margin = int(single_image_size/4)
        pandaimg = pandaimg[margin:-margin,margin:-margin,:]
        pandaimg = cv2.resize(pandaimg, (single_image_size, single_image_size), interpolation=cv2.INTER_AREA)

        robot_with_image_vertical = np.vstack((pandaimg, camera_image))
        concatenated_image = np.hstack((concatenated_image, robot_with_image_vertical))

    cv2.namedWindow(f'{name} Images', cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f'{name} Images', single_image_size*images, single_image_size*2) 
    cv2.imshow(f'{name} Images', concatenated_image)
    cv2.waitKey(0)  



def test_plot_cube(cube_center = [10.0,10.0,10.0], cube_width = 0.5):
    cw = cube_width
    cx = cube_center[0]
    cy = cube_center[1]
    cz = cube_center[2]
    points = np.array([ [cx-cw, cy-cw, cz-cw],
                        [cx+cw, cy-cw, cz-cw],
                        [cx+cw, cy+cw, cz-cw],
                        [cx-cw, cy+cw, cz-cw],
                        [cx-cw, cy-cw, cz+cw],
                        [cx+cw, cy-cw, cz+cw],
                        [cx+cw, cy+cw, cz+cw],
                        [cx-cw, cy+cw, cz+cw]])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    r = [-cw,cw]
    X, Y = np.meshgrid(r, r)
    ax.plot_surface(X+cx,Y                +cy,np.array([[+cw]])+cz, alpha=0.5)
    ax.plot_surface(X+cx,Y                +cy,np.array([[-cw]])+cz, alpha=0.5)
    ax.plot_surface(X+cx,np.array([[-cw]])+cy,Y                +cz, alpha=0.5)
    ax.plot_surface(X+cx,np.array([[+cw]])+cy,Y                +cz, alpha=0.5)
    ax.plot_surface(np.array([[+cw]])+cx,X+cy,Y+cz, alpha=0.5)
    ax.plot_surface(np.array([[-cw]])+cx,X+cy,Y+cz, alpha=0.5)
    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def plot_threshold_labelled(observations, labels):
    # Plotting
    plt.figure(figsize=(10, 5))  # Set the figure size

    # Scatter plot
    for label in np.unique(labels):
        # Select observations by label
        idx = labels == label
        plt.scatter(observations[idx], np.zeros_like(observations[idx]) + label,  # Adjust y-values to separate points vertically
                    c=['red' if label == 0 else 'blue'][0],  # Color red for label 0, blue for label 1
                    label=f'Label {label}')

    # Adding labels and title
    plt.xlabel('Observation Value')
    plt.ylabel('Label')
    plt.title('Observation Values and Labels')
    plt.yticks([0, 1])  # Set y-ticks to only show available labels

    # Add a legend
    plt.legend()

    # Show the plot
    plt.grid(True)  # Optional: Adds a grid for easier readability
    plt.show()


def plot_risk_data(data_door, data_peg):
    data_door = np.array(data_door)
    data_peg = np.array(data_peg)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2))  # Smaller plot size

    # Plot door data
    ax1.plot(data_door[:, 0], data_door[:, 1], marker='*', linestyle='-', linewidth=2, label='Door Risk')  
    ax1.axhline(0.5, color='blue', linestyle='--', linewidth=2, label='Risk Threshold (0.5)')
    ax1.set_xlabel('Door Opened [-]')
    ax1.set_ylabel('Risk Value')
    ax1.set_ylim(0, 1)
    ax1.grid(True)
    ax1.legend(loc='lower center')
    # Plot peg data
    ax2.plot(data_peg[:, 0], data_peg[:, 1], marker='*', linestyle='-', linewidth=2, label='Peg Risk')  
    ax2.axhline(0.5, color='blue', linestyle='--', linewidth=2, label='Risk Threshold (0.5)')
    ax2.set_xlabel('Peg Rotation ($^{\circ}$)')
    ax2.set_ylim(0, 1)
    ax2.grid(True)
    ax2.legend(loc='lower center')
    
    # Adjust layout and move legend to the bottom
    plt.tight_layout()
    plt.savefig("risk_plot.pdf")
    plt.show()

def plot_risk_polar(data_door, data_peg):
    # Convert lists to NumPy arrays
    data_door = np.array(data_door)
    data_peg = np.array(data_peg)

    # Convert door opening values into angles (mapping 0-1 range to 0-180 degrees)
    angles_door = np.linspace(0, np.pi, len(data_door[:, 0]))  # 0 to 180 degrees
    angles_peg = np.radians(data_peg[:, 0])  # Convert peg angles to radians

    # Extract risk values
    risk_door = data_door[:, 1]
    risk_peg = data_peg[:, 1]

    # Create polar subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 6))

    # Plot door data in polar coordinates
    ax1.plot(angles_door, risk_door, marker='o', linestyle='-', label='Door Risk')
    ax1.set_title("Risk vs. Door Opened (Polar)")
    ax1.set_theta_zero_location('N')  # Set 0 degrees at the top
    ax1.set_theta_direction(-1)  # Clockwise rotation
    ax1.set_rticks([0.2, 0.5, 0.8])  # Radial ticks
    ax1.set_ylim(0, 1)  # Set radius limit
    ax1.axhline(0.5, color='red', linestyle='--', label="Risk Threshold (0.5)")
    ax1.legend()

    # Plot peg data in polar coordinates
    ax2.plot(angles_peg, risk_peg, marker='o', linestyle='-', label='Peg Risk')
    ax2.set_title("Risk vs. Peg Rotation (Polar)")
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.set_rticks([0.2, 0.5, 0.8])
    ax2.set_ylim(0, 1)
    ax2.axhline(0.5, color='red', linestyle='--', label="Risk Threshold (0.5)")
    ax2.legend()

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
def plot_skill_data(data_peg_pick, data_peg_place, data_peg_door):
    skills = ['Peg Pick', 'Peg Place', 'Peg Door']
    metrics = ['Execution Success', 'GP (No Filtering)', 'GP (Filtering)', 'MLP']
    
    data = np.array([data_peg_pick, data_peg_place, data_peg_door])
    
    improvements = data[:, 2] - data[:, 0]
    
    # These are First round of results and will be changed
    correct_values = [
        [20, 30], [30+26, 60], [30+29, 60], [12, 100],  # Peg Pick
        [20, 30], [23+22, 60], [26+25, 60], [14, 100],  # Peg Place
        [18, 30], [30+27, 60], [23+30, 60], [15, 100]   # Peg Door
    ]
    
    x = np.arange(len(skills))
    width = 0.2  # width of each bar
    
    fig, ax = plt.subplots(figsize=(5, 3))
    
    # Plot main bars
    for i in range(len(metrics)):
        bars = ax.bar(x + (i - 1.5) * width, data[:, i], width, label=metrics[i])
        
        # Annotate bars with correct/total text
        for j, bar in enumerate(bars):
            value = correct_values[j * 4 + i]
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f"{value[0]}/{value[1]}", ha='center', fontsize=8)
    
    # Plot improvement bars (stacked on Execution Success)
    improvement_bars = ax.bar(x - 1.5 * width, improvements, width, bottom=data[:, 0], 
                               color='green', alpha=0.5, hatch='//', label='Improvement')

    # Annotate improvement bars
    for bar, imp in zip(improvement_bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_y() + 1, 
                f"+{imp:.1f}%", ha='center', fontsize=8, color='green')

    ax.set_ylabel('Percentage')
    ax.set_xticks(x)
    ax.set_xticklabels(skills)
    ax.set_ylim(0, 100)
    
    # Legend at bottom
    ax.legend(loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.3))
    
    plt.tight_layout()
    plt.show()

"""
Total: 90 exectutions, 180 risks - (out of these 90 demonstrations, the 9 were used for training) 
30 peg pick, 30 peg open door, 30 peg place demonstrations
30 

These are First round of results and will be changed
"""
data_peg_pick = [
    20.      / 30 * 100, # execution successfull
    (30.+26) / 60 * 100, # without filtering
    (30.+29) / 60 * 100, # with filtering
    (28.+28.)/ 60 * 100, # MLP
]
data_peg_place = [
    20.      / 30 * 100, # execution successfull
    (23.+22) / 60 * 100, # without filtering
    (26.+25) / 60 * 100, # with filtering
    (28.+28.)/ 60 * 100, # MLP
]
data_peg_door = [
    18.      / 30 * 100,
    (30.+23) / 60 * 100,
    (27.+30) / 60 * 100,
    (28.+28.)/ 60 * 100, # MLP
]

""" Data acquired from recorded video during experiment """
data_door = [
    [0.0, 0.386],
    [0.1, 0.418],
    [0.2, 0.428],
    [0.3, 0.433],
    [0.4, 0.435],
    [0.5, 0.453],
    [0.6, 0.521],
    [0.7, 0.661],
    [0.8, 0.814],
    [0.9, 0.839],
    [1.0, 0.839],
]
data_peg = [
    [-20, 0.48],
    [-10, 0.43],
    [0, 0.445],
    [15, 0.444],
    [30, 0.49],
    [45, 0.66],
    [55, 0.65],
    [95, 0.71],
    [105, 0.79],
    [120, 0.77],
]


if __name__ == '__main__':
    # Call the function to generate the plots
    # plot_risk_data(data_door, data_peg)
    # plot_risk_polar(data_door, data_peg)
    plot_skill_data(data_peg_pick, data_peg_place, data_peg_door)

    # img = get_panda_at_config(q=[0.,0.,0.,0.,0.,0.,0.])
    # print(img.shape)
    # test_plot_cube()