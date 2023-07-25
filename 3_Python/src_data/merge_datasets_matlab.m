close all;  clear all;  clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ToDo:    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% --- Settings
setOptions.do_2nd_run = false;
setOptions.doResort = true;
addon = "_Sorted";

% --- Settings für Martinez
%criterion_CheckDismiss = [3 0.7];
%criterion_Run0 = 0.98;
%criterion_Resort = 0.98;

% --- Settings für Quiroga
criterion_CheckDismiss = [2 0.96];
criterion_Run0 = 0.98;
criterion_Resort = 0.95;

%% --- Definition of Background Activity Detection (BAD)
setOptions.do_bad_dec = false;
bad_id = {0, 1, 2, 3};
bad_text = {"noise", "artefact", "background activity", "spontaneous activity"};

%% --- Vorverarbeitung: Datenaufnahme
% --- Daten laden
[filename, path2data] = uigetfile('../data/*.mat');

% --- Multiselect-Auswertung
setOptions.path2file = strcat(path2data, filename);
setOptions.path2save = strcat(path2data, filename(1:end-4), addon, ".mat");
setOptions.path2fig = strcat(path2data, filename(1:end-4));

load(setOptions.path2file);
clear filename addon path2data;
disp("Start of merging the data set to one");

%% --- Pre-Processing: Input structuring
if size(frames_cluster, 1) == 1
    frames_cluster = frames_cluster';
end

input_cluster = unique(frames_cluster);
data_0raw = cell(4, length(input_cluster));
data_0raw_number = 0;
for idx = 1:length(input_cluster)
    pos_in = find(frames_cluster == input_cluster(idx));
    data_0raw{1, idx} = input_cluster(idx);
    data_0raw{2, idx} = pos_in;
    data_0raw{3, idx} = double(frames_in(pos_in, :));
    data_0raw{4, idx} = mean(double(frames_in(pos_in, :)));

    YCheck = data_0raw{3, idx}; 
    WaveRef = crossval(mean(YCheck), mean(YCheck));
    for idy = 1:length(pos_in)
        WaveIn = crossval(YCheck(idy,:), mean(YCheck));     
        metric_Check0(idy, :) = calc_metric(WaveIn, WaveRef);
    end
    data_0raw{5, idx} = metric_Check0;
    data_0raw_number = data_0raw_number + length(pos_in);
end
clear idx pos_in metric_Check0 WaveIn WaveRef YCheck;
disp(" ... data are loaded and pre-selected");

%% --- Pre-Processing: Konsistenz-Prüfung der Frames pro Cluster (mit Mean)
% Korrelation, ob Mean-Waveform vom Cluster mit Frames übereinstimmen (Falsche rausfiltern)
data_1process = cell(5, length(input_cluster));
data_1dismiss = cell(5, length(input_cluster));

for idx = 1:1:length(input_cluster)
    do_run = true;
    YCheckIn = data_0raw{3, idx};
    XCheckIn = data_0raw{2, idx};
    XCheck = 1:length(XCheckIn);
    XCheck_False = [];
    check = [];

    IteNo = 0;
    while(do_run)
        XCheck(check) = [];                     
        % --- Crosscorrelation of each frame with mean frame
        metric_Check0 = [];
        YCheck = YCheckIn(XCheck, :);
        WaveRef = crossval(mean(YCheck), mean(YCheck));
        for idy = 1:1:length(XCheck)
            WaveIn = crossval(YCheck(idy,:), mean(YCheck));     
            metric_Check0(idy, :) = [calc_metric(WaveIn, WaveRef), XCheck(idy)];
        end
        % --- Decision for filtering
        check = find((abs(metric_Check0(:,1)) > criterion_CheckDismiss(1)) | (metric_Check0(:,5) < criterion_CheckDismiss(2)));
        XCheck_False = [XCheck_False; metric_Check0(check, 6)];
        if(~isempty(check) && (IteNo <= 100))
            do_run = true;
            IteNo = IteNo +1;
        else
            do_run = false;
            IteNo = IteNo;
        end                
    end

    % Übergabe: Processing frames
    metric_Check1 = [];
    YCheck = YCheckIn(XCheck, :); 
    WaveRef = crossval(mean(YCheck), mean(YCheck));
    for idy = 1:length(XCheck)
        WaveIn = crossval(YCheck(idy,:), mean(YCheck));     
        metric_Check1(idy, :) = calc_metric(WaveIn, WaveRef);
    end
    data_1process{1, idx} = input_cluster(idx);
    data_1process{2, idx} = [XCheckIn(XCheck), idx + ones(size(XCheckIn(XCheck)))];
    data_1process{3, idx} = YCheckIn(XCheck, :);
    data_1process{4, idx} = mean(YCheckIn(XCheck, :));
    data_1process{5, idx} = metric_Check1;

    % Übergabe: Dismissed frames
    metric_Check1 = [];
    YCheck = YCheckIn(XCheck_False, :); 
    WaveRef = crossval(mean(YCheck), mean(YCheck));
    for idy = 1:1:length(XCheck_False)
        WaveIn = crossval(YCheck(idy,:), mean(YCheck));     
        metric_Check1(idy, :) = calc_metric(WaveIn, WaveRef);
    end
    data_1dismiss{1, idx} = input_cluster(idx);
    data_1dismiss{2, idx} = [XCheckIn(XCheck_False), idx + ones(size(XCheckIn(XCheck_False)))];
    data_1dismiss{3, idx} = YCheckIn(XCheck_False, :);
    data_1dismiss{4, idx} = mean(YCheckIn(XCheck_False, :));
    data_1dismiss{5, idx} = metric_Check1;
end
clear do_run check YCheck YCheckIn XCheckIn XCheck XCheck_False Ymean WaveIn WaveRef;
clear idx idy IteNo metric_Check1;
disp(" ... End of step #1");

%% --- Processing: Merging cluster
data_2merge = data_1process(1:4, 1);
data_2wrong = cell(4, 1);

data_2merge_number = 1;
data_2wrong_number = 0;
data_missed_new = cell(5, 1);
for idx = 2:length(input_cluster)
    Yraw_New = data_1process{3, idx};
    Ymean_New = data_1process{4, idx};
    Xraw_New = data_1process{2, idx};    
    
    %--- Erste Prüfung: Mean-Waveform vergleichen mit bereits gemergten Clustern
    metric_Run0 = [];    
    for idy = 1:1:size(data_2merge, 2)
        Ycheck_Mean = data_2merge{4, idy};            
        WaveIn = crossval(Ymean_New, Ycheck_Mean); 
        WaveRef = crossval(Ycheck_Mean, Ycheck_Mean);
        metric_Run0(idy, :) = calc_metric(WaveIn, WaveRef);
    end
    % Entscheidung treffen
    [candY, candX] = max(metric_Run0(:, 5));
    if(isempty(candX)) 
        % Keine Lösung vorhanden --> Anhängen
        data_2wrong_number = data_2wrong_number + 1;
        data_2wrong{1, data_2wrong_number} = idx;
        data_2wrong{2, data_2wrong_number} = Xraw_New;
        data_2wrong{3, data_2wrong_number} = Yrew_New;
        data_2wrong{4, data_2wrong_number} = Ymean_New;
    elseif(candY >= criterion_Run0) 
        % Zweite Prüfung --> Einzel-Waveform mit Mean
        YCheck = [data_2merge{3, candX}; Yraw_New];
        XCheck = [data_2merge{2, candX}; Xraw_New];
        YMean = data_2merge{4, candX};
        
        WaveRef = crossval(YMean, YMean);
        metric_Run1 = [];
        for idz = 1:size(YCheck, 1)
            WaveIn = crossval(YCheck(idz, :), YMean);
            metric_Run1(idz, :) = calc_metric(WaveIn, WaveRef);
        end
        
        selOut = find(metric_Run1(:, 5) <= 0.92);
        if(~isempty(selOut))
            selCnt = length(selOut);
            data_missed_new{2, 1} = [data_missed_new{2, 1}; XCheck(selOut, :)];
            data_missed_new{3, 1} = [data_missed_new{3, 1}; YCheck(selOut, :)];
            
            XCheck(selOut, :) = [];
            YCheck(selOut, :) = [];
        end

        % Potentieller Match
        data_2merge_number = data_2merge_number;
        data_2merge{1, candX} = data_2merge{1, candX};
        data_2merge{2, candX} = XCheck;
        data_2merge{3, candX} = YCheck;
        data_2merge{4, candX} = mean(data_2merge{3, candX});       
    else 
        % Neues Cluster
        data_2merge_number = data_2merge_number + 1;
        data_2merge{1, data_2merge_number} = data_2merge_number-1;
        data_2merge{2, data_2merge_number} = Xraw_New;
        data_2merge{3, data_2merge_number} = Yraw_New;
        data_2merge{4, data_2merge_number} = Ymean_New;
    end
end
disp(" ... End of step #2");
data_missed_new{1,1} = size(data_1dismiss, 2) + 1;
data_missed_new{4,1} = mean(data_missed_new{3,1});
data_1dismiss = [data_1dismiss, data_missed_new];

for idx = 1:data_2merge_number
    metric_Run1 = [];
    Yraw_New = data_2merge{3, idx};
    WaveRef = crossval(data_2merge{4, idx}, data_2merge{4, idx});
    for idy = 1:1:size(Yraw_New, 1)
        WaveIn = crossval(Yraw_New(idy,:), data_2merge{4, idx});     
        metric_Run1(idy, :) = calc_metric(WaveIn, WaveRef);
    end
    data_2merge{5, idx} = metric_Run1;
end

clear idx idy idz candX candY selOut selCnt;
clear Yraw_New Ymean_New Xraw_New Ycheck_Mean WaveRef WaveIn;
clear candX candY SelCluster mode;
clear data_missed_new;

%% --- Post-Processing: Resorting dismissed frames
data_restored = 0;
if(setOptions.doResort)
    for idx0 = 1:size(data_1dismiss, 2)
        pos_sel = data_1dismiss{2, idx0};
        frames_sel = data_1dismiss{3, idx0};
        for idx1 = 1:size(frames_sel, 1)
            % --- Calculation of metric
            for idx2 = 1:data_2merge_number
                WaveRef = crossval(data_2merge{4, idx2}, data_2merge{4, idx2});
                WaveIn = crossval(frames_sel(idx1, :), data_2merge{4, idx2});     
                metric_Run2(idx2, :) = calc_metric(WaveIn, WaveRef);
            end
            % --- Decision
            [selY, selX] = max(metric_Run2(:,5));
            if(selY >= criterion_Resort)
                data_2merge{2, selX} = [data_2merge{2, selX}; pos_sel(idx1, :)];
                data_2merge{3, selX} = [data_2merge{3, selX}; frames_sel(idx1, :)];
                data_restored = data_restored + 1;
            end
        end
    end
    clear idx0 idx1 idx2 selX selY;
    clear frames_sel WaveRef WaveIn
    disp(" ... resorting dismissed frames to original");
end

%% --- Preparing: Transfer to new file
output.frames = [];
output.cluster = [];
data_process_num = 0;
for idx = 1:data_2merge_number
    X = double(data_2merge{1, idx}) .* ones(length(data_2merge{2, idx}), 1);
    Z = data_2merge{3, idx};
    
    if idx == 1
        output.cluster = X;
        output.frames = Z;
    else
        output.cluster = [output.cluster; X];
        output.frames = [output.frames; Z];
    end
    data_process_num = data_process_num + length(X);
end
clear idx X Z;
disp(" ... merged output generated");

%% --- Preparing: BAD Decision
cluster_pre = output.cluster;

frames_bad = 0 * cluster_pre;
if(setOptions.do_bad_dec)
    disp("Please make your decision:");
    for idx = 1:1:size(bad_id, 2)
        disp(strcat(num2str(bad_id{idx}), " = ", bad_text{idx}, ", "));
    end
       
    for idx = (1:1:size(unique(cluster_pre)))
        test_plot(output.frames, cluster_pre, idx-1);
        pause(2);
        sel_id = input("And?");
        
        sel = find(cluster_pre == idx-1);
        frames_bad(sel) = sel_id;
        close;
    end
end
clear idx cluster_pre sel_id sel;

%% --- Preparing: Plot results
plot_results(data_2merge, setOptions.path2fig, '_ResultsMerged_Fig');
plot_results(data_1dismiss, setOptions.path2fig, '_ResultsDismiss_Fig');

%% --- Saving output
data_ratio_merged = data_process_num / size(frames_in, 1);
data_ratio_dismiss = 1 - data_ratio_merged;

frames_in = output.frames;
frames_cluster = int16(output.cluster-1);

save(setOptions.path2save, 'frames_in', 'frames_cluster', 'frames_bad', 'data_ratio_merged');
clear FileName;

disp("This is the End!");

%% -------------------- EXTERNAL FUNCTIONS -------------------------------
function test_plot(frames_in, cluster, idx)
    Xsel = find(cluster == idx);
    frames_sel = frames_in(Xsel, :);
    mean0 = mean(frames_sel);

    figure();
    plot(frames_sel', 'k')
    hold on;
    plot(mean0, 'r')
end

function out = calc_metric(WaveIn, WaveRef)
    [Y1, X1] = max(WaveIn);
    [Y2, X2] = max(WaveRef);
    
    out = [];
    % --- Normal results
    out(1) = X1 - X2;
    out(2) = mse_loss(WaveIn, WaveRef);
    out(3) = abs(trapz(WaveIn(1:X1)) - trapz(WaveIn(X1:end)));
    out(4) = X1;
    out(5) = Y1;
end

function out = crossval(Wave1, Wave2)
    out = xcorr(Wave1, Wave2)/(sqrt(sum(Wave1.^2))* sqrt(sum(Wave2.^2)));
end

function outdB = calculate_snr(wave_in, wave_mean)    
    A = (max(wave_mean) - min(wave_mean)).^2;
    B = sum((wave_in - wave_mean).^2);
    outdB = 10*log10(A/B); 
end

function val = mse_loss(yin, yref)
    val = sum((yin - yref).^2);
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
