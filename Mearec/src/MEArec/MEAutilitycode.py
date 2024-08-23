

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
def plot_electrode_positions(mea):
    plt.figure(figsize=(6, 6))
    plt.plot(mea.positions[:, 1], mea.positions[:, 2], 'b*')
    plt.plot(mea.positions[0, 1], mea.positions[0, 2], 'r*')  # first electrode
    plt.plot(mea.positions[9, 1], mea.positions[9, 2], 'g*')  # tenth electrode
    plt.plot(mea.positions[10, 1], mea.positions[10, 2], 'y*') # eleventh electrode
    plt.plot(mea.positions[-1, 1], mea.positions[-1, 2], 'c*') # last electrode
    plt.axis('equal')
    plt.title("Electrode Positions")
    plt.show()

# 2D Probe Plot
def plot_probe(mea):
    MEA.plot_probe(mea)

# Moving the MEA
def move_mea(mea):
    plt.figure(figsize=(6, 6))
    plt.plot(mea.positions[:, 1], mea.positions[:, 2], 'b*')
    mea.move([0, 50, 50])  # Move all electrodes by [0, 50, 50]
    plt.plot(mea.positions[:, 1], mea.positions[:, 2], 'r*')
    mea.move([0, -300, 0])  # Move all electrodes by [0, -300, 0]
    plt.plot(mea.positions[:, 1], mea.positions[:, 2], 'g*')
    plt.axis('equal')
    plt.title("MEA Movement")
    plt.show()

# Rotating the MEA
def rotate_mea(mea):
    plt.figure(figsize=(6, 6))
    plt.plot(mea.positions[:, 1], mea.positions[:, 2], 'b*')
    mea.rotate([1, 0, 0], 45)  # Rotate 45 degrees around the x-axis
    plt.plot(mea.positions[:, 1], mea.positions[:, 2], 'r*')
    plt.axis('equal')
    plt.title("MEA Rotation")
    plt.show()

    # Resetting the MEA to its initial position
    mea.rotate([1, 0, 0], -45)  # Rotate -45 degrees around the x-axis to reset
    plt.figure(figsize=(6, 6))
    plt.plot(mea.positions[:, 1], mea.positions[:, 2], 'b*')
    plt.axis('equal')
    plt.title("MEA Reset Position")
    plt.show()

# Handling currents
def handle_currents(mea):
    print("Initial Currents:\n", mea.currents)
    curr = np.arange(mea.number_electrodes)
    mea.currents = curr
    print("Currents After Setting:\n", mea.currents)
    mea.reset_currents()
    print("Currents After Resetting to 0:\n", mea.currents)
    mea.reset_currents(100)
    print("Currents After Resetting to 100:\n", mea.currents)

    # Simulating random currents
    mea.set_random_currents(mean=1000, sd=50)
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
def plot_potential_images(mea):
    mea.points_per_electrode = 1
    mea.reset_currents()
    mea[0][0].current = 10000
    mea[5][0].current = 10000
    mea[0][7].current = 10000
    _ = MEA.plot_v_image(mea, y_bound=[-100, 100], z_bound=[-100, 100], plane='yz', offset=10)

    fig, axes = plt.subplots(1, 2)
    mea.points_per_electrode = 1
    _, v1 = MEA.plot_v_image(mea, y_bound=[-55, -80], z_bound=[-55, -80], offset=2,
                             npoints=30, plane='yz', ax=axes[0])
    mea.points_per_electrode = 100
    _, v100 = MEA.plot_v_image(mea, y_bound=[-55, -80], z_bound=[-55, -80], offset=2,
                               npoints=30, plane='yz', ax=axes[1])

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    mea.points_per_electrode = 1
    _ = MEA.plot_v_surf(mea, v_plane=v1, y_bound=[-55, -80], z_bound=[-55, -80], offset=10,
                        npoints=30, plane='yz', ax=ax1)
    _ = MEA.plot_v_surf(mea, v_plane=v100, y_bound=[-55, -80], z_bound=[-55, -80], offset=10,
                        npoints=30, plane='yz', ax=ax2)

    mea.points_per_electrode = 1
    mea[0][0].current = 10000
    ax, v = MEA.plot_v_surf(mea, y_bound=[-100, 100], z_bound=[-100, 100],
                            plane='yz', plot_plane='yz', offset=30, distance=200)
    MEA.plot_probe_3d(mea, ax=ax, xlim=[-500, 500], color_currents=True)

    mea.rotate([0, 1, 0], 90)
    ax, v = MEA.plot_v_surf(mea, x_bound=[-100, 100], y_bound=[-100, 100],
                            plane='xy', plot_plane='xy', offset=30, distance=30)
    MEA.plot_probe_3d(mea, ax=ax, xlim=[-100, 100], zlim=[-100, 300], color_currents=True, type='planar')

    mea.rotate([0, 1, 0], -90)
    mea.rotate([0, 0, 1], -90)
    ax, v = MEA.plot_v_surf(mea, x_bound=[-100, 100], z_bound=[-100, 100],
                            plane='xz', plot_plane='xz', offset=30, distance=100)
    MEA.plot_probe_3d(mea, ax=ax, color_currents=True)

# Simulating noise signal and plotting
def plot_noise_signal(mea):
    signals = np.random.randn(mea.number_electrodes, 10000)
    _ = MEA.plot_mea_recording(signals, mea, lw=0.1)
    plt.title('Simulated Signal Traces')
    plt.xlabel('Time')
    plt.ylabel('Signal Amplitude')
    plt.show()

# Adding a user-defined MEA from YAML
def add_user_defined_mea(yaml_path):
    try:
        with open(yaml_path, 'r') as file:
            user_info = yaml.safe_load(file)

        print("User-defined MEA Info:\n", user_info)

        user_positions = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
            [40.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
            [60.0, 0.0, 0.0],
            [70.0, 0.0, 0.0],
            [80.0, 0.0, 0.0],
            [90.0, 0.0, 0.0]
            # Add more positions as needed
        ])

        # Construct the user MEA with positions
        user_mea = MEA.MEA(info=user_info, positions=user_positions)

        # Output some properties to verify
        print("User MEA Object Type:", type(user_mea))
        print("User MEA Number of Electrodes:", user_mea.number_electrodes)

        return user_mea

    except FileNotFoundError:
        print(f"Error: File '{yaml_path}' not found. Please check the file path.")
        return None

# Main function to call all operations
def main():
    list_meas()
    mea_name = 'SqMEA-10-15'
    load_mea_info(mea_name)
    mea = instantiate_mea(mea_name)
    plot_electrode_positions(mea)
    plot_probe(mea)
    move_mea(mea)
    rotate_mea(mea)
    handle_currents(mea)
    plot_probe_3d(mea)
    plot_potential_images(mea)
    plot_noise_signal(mea)

 #   yaml_path = '/path/to/your/user_mea.yaml'
 #   user_mea = add_user_defined_mea(yaml_path)
 #   plot_probe(user_mea)

if __name__ == "__main__":
    main()
