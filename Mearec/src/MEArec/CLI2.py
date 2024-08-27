import MEArec as mr
import os

# Specify the path to the cell models folder
cell_models_folder = "/home/mahmoud/MEArec/src/MEArec/cell_models/bbp"

# Generate templates
tempgen = mr.gen_templates(
    cell_models_folder=cell_models_folder,
    # Add other template generation parameters here...
)

# Save the generated templates
templates_file = 'generated_templates.h5'
mr.save_template_generator(tempgen, filename=templates_file)
print("Templates generated and saved successfully.")

# Specify the path to save the recordings
output_file = 'output_recordings.h5'

# Generate recordings
recgen = mr.gen_recordings(
    templates=tempgen,
    cell_models_folder=cell_models_folder,
    # Add other recording generation parameters here...
)

# Save the generated recordings
mr.save_recording_generator(recgen, filename=output_file)
print(f"Recordings saved to {output_file}")
