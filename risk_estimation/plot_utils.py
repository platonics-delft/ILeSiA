
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
        camera_image = cv2.resize(camera_image, (single_image_size,single_image_size))

        panda_config = robot_states[idx]
        pandaimg = get_panda_at_config(q=panda_config)
        # pandaimg = cv2.cvtColor(pandaimg, cv2.COLOR_BGR2GRAY)
        margin = int(single_image_size/4)
        pandaimg = pandaimg[margin:-margin,margin:-margin,:]
        pandaimg = cv2.resize(pandaimg, (single_image_size, single_image_size))

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

if __name__ == '__main__':
    img = get_panda_at_config(q=[0.,0.,0.,0.,0.,0.,0.])
    print(img.shape)
    test_plot_cube()