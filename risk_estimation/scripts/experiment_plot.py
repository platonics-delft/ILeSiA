
import numpy as np
import matplotlib.pyplot as plt

def plot_risk_data(data_door, data_peg):
    """ Const. plot from experiment """
    data_door = np.array(data_door)
    data_peg = np.array(data_peg)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))  # Smaller plot size

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
    """ Const. plot from experiment """
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

def plot_skill_data(data_):
    """ Const. plot from experiment """
    skills = ['Peg Pick', 'Peg Place', 'Door Open']
    metrics = ['Execution Success', '$\mathcal{GP}$', '$\mathcal{GP}$ (Sliding window)', '$\mathcal{MLP}$']
    
    skills_data = []
    for skill in data_:
        skills_data.append([100*suc/all for suc, all in skill])
    data = np.array(skills_data)
    # data = np.array([data_peg_pidata_doorck, data_peg_place, data_peg_door])
    
    improvements = data[:, 3] - data[:, 0]
    
    x = np.arange(len(skills))
    width = 0.2  # width of each bar
    
    fig, ax = plt.subplots(figsize=(6, 3))
    
    # Plot main bars
    for i in range(len(metrics)):
        bars = ax.bar(x + (i - 1.5) * width, data[:, i], width, label=metrics[i])
        
        # Annotate bars with correct/total text
        for j, bar in enumerate(bars):
            value = data_[j][i]
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f"{value[0]}/{value[1]}", ha='center', fontsize=8)
    
    # Plot improvement bars (stacked on Execution Success)
    improvement_bars = ax.bar(x + 1.5 * width, improvements, width, bottom=data[:, 0], 
                               color='green', alpha=0.5, hatch='//', label='Improvement')

    # Annotate improvement bars
    for bar, imp in zip(improvement_bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_y() + 5, 
                f"+{imp:.1f}%", ha='center', fontsize=8, color='green')

    ax.grid()
    ax.set_ylabel('Accuracy [$\%$]')
    ax.set_xticks(x)
    ax.set_xticklabels(skills)
    ax.set_ylim(0, 100)
    
    # Legend at bottom
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    # plt.show()
    plt.savefig("skill_plot.pdf")

"""
Total: 90 exectutions, 180 risks - (out of these 90 demonstrations, the 9 were used for training) 
30 peg pick, 30 peg open door, 30 peg place demonstrations
30 

These are First round of results and will be changed
"""
skill_data = [
    # exec.s, MLP      , GP         , GP slid.w. , MLP
    [[20, 30], [44, 60], [57, 60], [59, 60]],  # Peg Pick
    [[20, 30], [44, 60], [47, 60], [49, 60]],  # Peg Place
    [[18, 30], [35, 60], [54, 60], [57, 60]],  # Peg Door
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
    plot_risk_data(data_door, data_peg)
    # plot_risk_polar(data_door, data_peg)
    # plot_skill_data(skill_data)

    # print(img.shape)