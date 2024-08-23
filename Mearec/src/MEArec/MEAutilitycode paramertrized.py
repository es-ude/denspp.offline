import MEAutility as MEA
import matplotlib.pyplot as plt
import numpy as np
import yaml

# List available MEAs
def list_meas():
    available_meas = MEA.return_mea()
    print("Available MEA:\n", available_meas)

# Load and inspect a specific MEA model
def load_mea_info(mea_name):
    mea_info = MEA.return_mea_info(mea_name)
    print(f"{mea_name} Info:\n", mea_info)

# Instantiate the MEA object
def instantiate_mea(mea_name):
    mea = MEA.return_mea(mea_name)
    print("MEA Object Type:", type(mea))
    print("Number of Electrodes:", mea.number_electrodes)
    print("Dimensions:", mea.dim)
    return mea

# 2D Plot of electrode positions
def plot_electrode_positions(mea, electrodes_to_highlight):
    plt.figure(figsize=(6, 6))
    plt.plot(mea.positions[:, 1], mea.positions[:, 2], 'b*')
    colors = ['r*', 'g*', 'y*', 'c*']
    for i, elec in enumerate(electrodes_to_highlight):
        plt.plot(mea.positions[elec, 1], mea.positions[elec, 2], colors[i % len(colors)])
    plt.axis('equal')
    plt.title("Electrode Positions")
    plt.show()

# 2D Probe Plot
def plot_probe(mea):
    MEA.plot_probe(mea)

# Moving the MEA
def move_mea(mea, movements):
    plt.figure(figsize=(6, 6))
    plt.plot(mea.positions[:, 1], mea.positions[:, 2], 'b*')
    colors = ['r*', 'g*']
    for i, move in enumerate(movements):
        mea.move(move)
        plt.plot(mea.positions[:, 1], mea.positions[:, 2], colors[i % len(colors)])
    plt.axis('equal')
    plt.title("MEA Movement")
    plt.show()

# Rotating the MEA
def rotate_mea(mea, rotations):
    plt.figure(figsize=(6, 6))
    plt.plot(mea.positions[:, 1], mea.positions[:, 2], 'b*')
    for axis, angle in rotations:
        mea.rotate(axis, angle)
        plt.plot(mea.positions[:, 1], mea.positions[:, 2], 'r*')
    plt.axis('equal')
    plt.title("MEA Rotation")
    plt.show()

    # Resetting the MEA to its initial position
    for axis, angle in reversed(rotations):
        mea.rotate(axis, -angle)
    plt.figure(figsize=(6, 6))
    plt.plot(mea.positions[:, 1], mea.positions[:, 2], 'b*')
    plt.axis('equal')
    plt.title("MEA Reset Position")
    plt.show()

# Handling currents
def handle_currents(mea, initial_currents, reset_values, random_current_params):
    print("Initial Currents:\n", mea.currents)
    curr = np.arange(mea.number_electrodes)
    mea.currents = curr
    print("Currents After Setting:\n", mea.currents)
    
    for reset_value in reset_values:
        mea.reset_currents(reset_value)
        print(f"Currents After Resetting to {reset_value}:\n", mea.currents)
    
    mean, sd = random_current_params
    mea.set_random_currents(mean=mean, sd=sd)
    print("Random Currents:\n", mea.currents)
    plt.hist(mea.currents, bins=15)
    plt.title("Histogram of Random Currents")
    plt.show()

# 3D Probe Plot
def plot_probe_3d(mea):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = mea.positions[:, 0]
    y = mea.positions[:, 1]
    z = mea.positions[:, 2]
    ax.scatter(x, y, z, c='b', marker='o')
    ax.set_xlabel('X (microns)')
    ax.set_ylabel('Y (microns)')
    ax.set_zlabel('Z (microns)')
    ax.set_title('3D Probe Plot')
    plt.show()

# Electric potential images
def plot_potential_images(mea, currents, y_bounds, z_bounds, plane, offsets, npoints_list):
    mea.points_per_electrode = 1
    mea.reset_currents()
    for elec, current in currents:
        mea[elec[0]][elec[1]].current = current
    _ = MEA.plot_v_image(mea, y_bound=y_bounds, z_bound=z_bounds, plane=plane, offset=offsets[0])

    fig, axes = plt.subplots(1, 2)
    mea.points_per_electrode = 1
    _, v1 = MEA.plot_v_image(mea, y_bound=y_bounds, z_bound=z_bounds, offset=offsets[1],
                             npoints=npoints_list[0], plane=plane, ax=axes[0])
    mea.points_per_electrode = 100
    _, v100 = MEA.plot_v_image(mea, y_bound=y_bounds, z_bound=z_bounds, offset=offsets[1],
                               npoints=npoints_list[1], plane=plane, ax=axes[1])

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    mea.points_per_electrode = 1
    _ = MEA.plot_v_surf(mea, v_plane=v1, y_bound=y_bounds, z_bound=z_bounds, offset=offsets[2],
                        npoints=npoints_list[0], plane=plane, ax=ax1)
    _ = MEA.plot_v_surf(mea, v_plane=v100, y_bound=y_bounds, z_bound=z_bounds, offset=offsets[2],
                        npoints=npoints_list[1], plane=plane, ax=ax2)

    mea.points_per_electrode = 1
    mea[0][0].current = 10000
    ax, v = MEA.plot_v_surf(mea, y_bound=y_bounds, z_bound=z_bounds,
                            plane=plane, plot_plane=plane, offset=offsets[3], distance=200)
    MEA.plot_probe_3d(mea, ax=ax, xlim=[-500, 500], color_currents=True)

    mea.rotate([0, 1, 0], 90)
    ax, v = MEA.plot_v_surf(mea, x_bound=[-100, 100], y_bound=[-100, 100],
                            plane='xy', plot_plane='xy', offset=offsets[4], distance=30)
    MEA.plot_probe_3d(mea, ax=ax, xlim=[-100, 100], zlim=[-100, 300], color_currents=True, type='planar')

    mea.rotate([0, 1, 0], -90)
    mea.rotate([0, 0, 1], -90)
    ax, v = MEA.plot_v_surf(mea, x_bound=[-100, 100], z_bound=[-100, 100],
                            plane='xz', plot_plane='xz', offset=offsets[5], distance=100)
    MEA.plot_probe_3d(mea, ax=ax, color_currents=True)

# Simulating noise signal and plotting
def plot_noise_signal(mea, num_samples):
    signals = np.random.randn(mea.number_electrodes, num_samples)
    _ = MEA.plot_mea_recording(signals, mea, lw=0.1)
    plt.title('Simulated Signal Traces')
    plt.xlabel('Time')
    plt.ylabel('Signal Amplitude')
    plt.show()

# Adding a user-defined MEA from YAML
def add_user_defined_mea(yaml_path, user_positions):
    try:
        with open(yaml_path, 'r') as file:
            user_info = yaml.safe_load(file)

        print("User-defined MEA Info:\n", user_info)

        user_positions = np.array(user_positions)

        # Construct the user MEA with positions
        user_mea = MEA.MEA(info=user_info, positions=user_positions)

        # Output some properties to verify
        print("User MEA Object Type:", type(user_mea))
        print("User MEA Number of Electrodes:", user_mea.number_electrodes)

        return user_mea

    except FileNotFoundError:
        print(f"Error: File '{yaml_path}' not found. Please check the file path.")
        return None

# Main function 
if __name__ == "__main__":

    list_meas()
    mea_name = 'SqMEA-10-15'
    load_mea_info(mea_name)
    mea = instantiate_mea(mea_name)
    
    electrodes_to_highlight = [0, 9, 10, -1]
    plot_electrode_positions(mea, electrodes_to_highlight)
    
    plot_probe(mea)
    
    movements = [[0, 50, 50], [0, -300, 0]]
    move_mea(mea, movements)
    
    rotations = [([1, 0, 0], 45)]
    rotate_mea(mea, rotations)
    
    initial_currents = np.arange(mea.number_electrodes)
    reset_values = [0, 100]
    random_current_params = (1000, 50)
    handle_currents(mea, initial_currents, reset_values, random_current_params)
    
    plot_probe_3d(mea)
    
    currents = [([0, 0], 10000), ([5, 0], 10000), ([0, 7], 10000)]
    y_bounds = [-100, 100]
    z_bounds = [-100, 100]
    plane = 'yz'
    offsets = [10, 2, 10, 30, 30, 30]
    npoints_list = [30, 30]
    plot_potential_images(mea, currents, y_bounds, z_bounds, plane, offsets, npoints_list)
    
    num_samples = 10000
    plot_noise_signal(mea, num_samples)

#   yaml_path = '/path/to/your/user_mea.yaml'
#   user_positions = [
#       [0.0, 0.0, 0.0],
#       [10.0, 0.0, 0.0],
#       [20.0, 0.0, 0.0],
#       [30.0, 0.0, 0.0],
#       [40.0, 0.0, 0.0],
#       [50.0, 0.0, 0.0],
#       [60.0, 0.0, 0.0],
#       [70.0, 0.0, 0.0],
#       [80.0, 0.0, 0.0],
#       [90.0, 0.0, 0.0]
#       # Add more positions as needed
#   ]
#   user_mea = add_user_defined_mea(yaml_path, user_positions)
#   plot_probe(user_mea)

