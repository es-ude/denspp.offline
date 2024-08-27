import MEArec as mr
import tempfile
import os

# Load the generated templates
templates_file = 'generated_templates2.h5'
loaded_templates = mr.load_templates(templates_file)

# Save the loaded templates to a temporary file
temp_dir = tempfile.mkdtemp()
temp_templates_file = os.path.join(temp_dir, 'temp_templates2.h5')
mr.save_template_generator(loaded_templates, filename=temp_templates_file)

# Define the parameters
params = {
    'spiketrains': {
        'n_exc': 2,
        'n_inh': 1,
        'f_exc': 5,
        'f_inh': 15,
        'st_exc': 1,
        'st_inh': 3,
        'min_rate': 0.5,
        'ref_per': 2,
        'process': 'poisson',
        'gamma_shape': 2,
        't_start': 0,
        'duration': 10
    },
    'cell_types': {
        'excitatory': ['STPC', 'TTPC1', 'TTPC2', 'UTPC'],
        'inhibitory': ['BP', 'BTC', 'ChC', 'DBC', 'LBC', 'MC', 'NBC', 'NGC', 'SBC']
    },
    'templates': {
        'file_path': temp_templates_file  # Pass the path to the temporary file
    },
    'recordings': {
        'fs': None,
        'sync_rate': 0,
        'sync_jitt': 1,
        'modulation': 'electrode',
        'sdrand': 0.05,
        'bursting': True,
        'exp_decay': 0.1,
        'n_burst_spikes': 10,
        'max_burst_duration': 100,
        'shape_mod': True,
        'shape_stretch': 30.0,
        'n_bursting': 3,
        'chunk_duration': 20,
        'noise_level': 0,
        'noise_mode': 'uncorrelated',
        'noise_color': False,
        'noise_half_distance': 30,
        'far_neurons_n': 300,
        'far_neurons_max_amp': 10,
        'far_neurons_noise_floor': 0.5,
        'far_neurons_exc_inh_ratio': 0.8,
        'color_peak': 500,
        'color_q': 1,
        'random_noise_floor': 1,
        'filter': True,
        'filter_cutoff': [300, 6000],
        'filter_order': 3,
        'overlap': False,
        'extract_waveforms': False
    },
    'drifting': {
        'drifting': False,
        'n_drifting': None,
        'preferred_dir': [0, 0, 1],
        'angle_tol': 15,
        'drift_mode_speed': 'slow',
        'drift_mode_probe': 'rigid',
        'drift_fs': 100,
        'non_rigid_gradient_mode': 'linear',
        'non_rigid_linear_min_factor': 0.5,
        'non_rigid_linear_direction': 1,
        'non_rigid_step_depth_boundary': None,
        'non_rigid_step_factors': None,
        'slow_drift_velocity': 5,
        'slow_drift_amplitude': None,
        'slow_drift_waveform': 'triangluar',
        'fast_drift_period': 10,
        'fast_drift_max_jump': 20,
        'fast_drift_min_jump': 5,
        't_start_drift': 0,
        't_end_drift': None
    },
    'seeds': {
        'spiketrains': None,
        'templates': None,
        'convolution': None,
        'noise': None
    }
}

# Generate recordings using the loaded templates
recgen = mr.gen_recordings(params=params, templates=temp_templates_file, n_jobs=-1, verbose=True)

# Save the generated recordings
output_file = 'output_recordings.h5'
mr.save_recording_generator(recgen, filename=output_file)

print(f"Recordings saved to {output_file}")

# Clean up temporary directory
import shutil
shutil.rmtree(temp_dir)