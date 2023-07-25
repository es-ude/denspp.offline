close all;  clear all;  clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ToDo:    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% --- Settings
addon = "_Sorted";


%% --- Vorverarbeitung: Datenaufnahme
% --- Daten laden
[filename, path2data] = uigetfile('../data/*_Sorted.mat');

% --- Multiselect-Auswertung
setOptions.path2file = strcat(path2data, filename);
setOptions.path2save = strcat(path2data, filename(1:end-4), addon, ".mat");
setOptions.path2fig = strcat(path2data, filename(1:end-4));

load(setOptions.path2file);
clear filename addon path2data;
disp("Start of merging the data set to one");

%% --- Pre-Processing: Input structuring
input_cluster = unique(frames_cluster);
data_0raw = cell(4, length(input_cluster));
data_0raw_number = 0;
for idx = 1:length(input_cluster)
    pos_in = find(frames_cluster == input_cluster(idx));
    data_0raw{1, idx} = input_cluster(idx);
    data_0raw{2, idx} = pos_in;
    data_0raw{3, idx} = double(frames_in(pos_in, :));
    data_0raw{4, idx} = mean(double(frames_in(pos_in, :)));

    data_0raw_number = data_0raw_number + length(pos_in);
end
clear idx pos_in;
disp(" ... data are loaded and pre-selected");

%% --- Preparing: Plot results
plot_results(data_0raw, setOptions.path2fig, '_ResultsMerged_Fig');

%% -------------------- EXTERNAL FUNCTIONS -------------------------------
function outdB = calculate_snr(wave_in, wave_mean)    
    A = (max(wave_mean) - min(wave_mean)).^2;
    B = sum((wave_in - wave_mean).^2);
    outdB = 10*log10(A/B); 
end

function val_snr = plot_results(data_packet, path2fig, name) 
    % --- SNR calculation
    for idx = 1:size(data_packet, 2)
        selX = data_packet{2, idx};
        selY = data_packet{3, idx};
        selM = data_packet{4, idx};
    
        val_snr = zeros(1, length(selX));
        for idy = 1:length(selX)
            val_snr(idy) = calculate_snr(selY(idy,:), selM);
        end
    end

    sizeID = size(data_packet, 2);
    if(sizeID >= 9)
            SubFig = [2, 6];
    else
        if(sizeID <= 4)
            SubFig = [1, 6];
        else
            SubFig = [2, 6];
        end
    end
    noSubFig = SubFig(1)* SubFig(2);
    for idy = 1:1:ceil(sizeID/noSubFig)
        figure('doublebuffer', 'off', 'visible', 'off');
        set(gcf, 'units','normalized','outerposition',[0 0 0.98 1]);
            
        selID = (1:1:noSubFig) + noSubFig*(idy-1);
        if(selID(end) > sizeID)
            selID = selID(1):1:sizeID;
        end
    
        IteNo = 1;
        for idx = 1:1:length(selID)
            NoCluster = selID(idx);
            
            % Decision, if more than 100 frames
            Yin = data_packet{3, NoCluster};
            if(size(Yin,1) > 2000)
                selFrames = randperm(size(Yin,1), 2000);
            else
                selFrames = 1:size(Yin,1);
            end

            val_snr0 = [];
            for idz = 1:size(Yin,1)
                val_snr0(idz) = calculate_snr(Yin(idz,:), mean(Yin));
            end
            
            % Plot
            formatSpec = '%.2f';
            subplot(SubFig(1), SubFig(2), IteNo);
            IteNo = IteNo +1;
            
            plot(Yin(selFrames,:)', 'b');
            hold on; grid on;
            plot(mean(Yin), 'r', 'Linewidth', 2);
    
            t = title([strcat("Cluster ID: ", num2str(NoCluster)), strcat("SNR = ", num2str(mean(val_snr0), formatSpec), " (\pm", num2str(std(val_snr0), formatSpec), ") dB - No = ", num2str(size(Yin,1)))]);
            set(gca, 'FontSize', 12);
            t.FontSize = 10;
        end        
        saveas(gcf, strcat(path2fig, name, num2str(idy, '%02d'), '.jpg'));
    end
end
