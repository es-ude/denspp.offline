import MEArec as mr

def generate_templates(cell_models_folder, template_params=None, templates_tmp_folder=None, intraonly=False, parallel=True, recompile=False, n_jobs=None, delete_tmp=True, verbose=False):
    # Generate templates using the provided parameters
    tempgen = mr.gen_templates(
        cell_models_folder,
        params=template_params,
        templates_tmp_folder=templates_tmp_folder,
        intraonly=intraonly,
        parallel=parallel,
        recompile=recompile,
        n_jobs=n_jobs,
        delete_tmp=delete_tmp,
        verbose=verbose
    )
    return tempgen

def save_templates(tempgen, filename):
    # Save the TemplateGenerator object to a file
    mr.save_template_generator(tempgen, filename=filename)


if __name__ == "__main__":
    
    # Default cell models folder (can be customized)
    cell_models_folder = mr.get_default_cell_models_folder()

    # Custom template generation parameters (default if None)
    template_params = {
        'sim_time': 1,
        'target_spikes': [3, 50],
        'cut_out': [2, 5],
        'dt': 0.03125,
        'delay': 10,
        'weights': [0.25, 1.75],
        'rot': 'physrot',
        'probe': 'Neuronexus-32',
        'ncontacts': 1,
        'overhang': 30,
        'offset': 0,
        'xlim': [10, 80],
        'ylim': None,
        'zlim': None,
        'det_thresh': 30,
        'n': 50,
        'seed': None,
        'drifting': False,
        'max_drift': 100,
        'min_drift': 30,
        'drift_steps': 50,
        'drift_xlim': [-10, 10],
        'drift_ylim': [-10, 10],
        'drift_zlim': [20, 80]
    }

    # Generate templates
    tempgen = generate_templates(cell_models_folder, template_params)

    # Save the generated templates to a file
    save_templates(tempgen, filename='generated_templates2.h5')

    print("Templates generated and saved successfully.")

