import subprocess

# Default paths (adjust as per your actual paths)
cell_models_folder = '/home/mahmoud/MEArec/src/MEArec/cell_models/bbp'
recordings_folder = '/home/mahmoud/MEArec/src/MEArec/output/recordings'
recordings_params_file = '/home/mahmoud/MEArec/src/MEArec/recordings_params.yaml'
templates_folder = '/home/mahmoud/MEArec/src/MEArec/output/templates'
templates_params_file = '/home/mahmoud/MEArec/src/MEArec/templates_params.yaml'

def set_default_config():
    """ Set default configurations for MEArec """
    commands = [
        f"mearec set-cell-models-folder {cell_models_folder}",
        f"mearec set-recordings-folder {recordings_folder}",
        f"mearec set-recordings-params {recordings_params_file}",
        f"mearec set-templates-folder {templates_folder}",
        f"mearec set-templates-params {templates_params_file}"
    ]
    
    for command in commands:
        subprocess.run(command, shell=True)

def generate_templates():
    """ Generate templates using MEArec """
    command = "mearec gen-templates --njobs 4"
    subprocess.run(command, shell=True)

def generate_recordings(templates_folder):
    """ Generate recordings using MEArec """
    command = f"mearec gen-recordings --templates {templates_folder}"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(e)

def main():
    set_default_config()
    generate_templates()
    generate_recordings()

if __name__ == "__main__":
    main()
