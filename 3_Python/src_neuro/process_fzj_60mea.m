close all; clear all; clc;

%
% MATLAB Tool for Pre-Processing the MCS datasets from FZ Jülich
% Make it ready for using in Python
%

% --- Loading the mat table
path2data = uigetdir(pwd);
file_name_mat = dir(fullfile(path2data, '*.mat'));

for idx = 1:1:length(file_name_mat)
    file_name = file_name_mat(idx).name(1:end-4);
    % Do not process if merged is available!
    if ~strcmp(file_name(end-6:end), '_merged') 
        load_file = strcat(file_name_mat(idx).folder, '\', file_name_mat(idx).name);
        data = load(load_file);
    
        %% --- Processing
        data0 = struct2cell(data);
        clear data;
        data0 = data0{1};
        
        time = table2array(data0(:,1))';
        gain = single(1e-6);
        electrode = single(table2array(data0(:, 2:end)));
        fs = uint16(1e3 / mean(diff(time)));
        head_name = data0.Properties.VariableNames;
        head_name(1) = []; 
        
        %% --- Saving
        file_name_new = strcat(file_name, '_merged.mat');
        save(file_name_new, 'gain', 'fs', 'head_name', 'electrode', '-v7.3');
    end
end
    
%% --- Finish
disp('DONE')
clear all;
