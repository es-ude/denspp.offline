import MEArec as mr
import matplotlib.pyplot as plt

# Define the path to the recordings file
recordings_path = '/home/mahmoud/MEArec/output_recordings.h5'

# Load recordings using MEArec
try:
    recgen = mr.load_recordings(recordings_path)
    print("Recordings loaded successfully")

    # Plot recordings
    try:
        mr.plot_recordings(recgen)
        plt.show()  # Explicitly show the plot
        print("Recordings plotted successfully")
    except Exception as e:
        print(f"Error plotting recordings: {e}")

except Exception as e:
    print(f"Error loading recordings: {e}")


# Define the path to the templates file
templates_path = '/home/mahmoud/MEArec/generated_templates.h5'

# Load templates using MEArec
try:
    tempgen = mr.load_templates(templates_path)
    print("Templates loaded successfully")

    # Plot templates
    try:
        mr.plot_templates(tempgen, template_ids=100, drifting=True)
        plt.show()  # Explicitly show the plot
        print("Templates plotted successfully")
    except Exception as e:
        print(f"Error plotting templates: {e}")

except Exception as e:
    print(f"Error loading templates: {e}")
