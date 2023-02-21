close all;  clear all;  clc;

%% --- Settings
do_plot = 1;
do_save = 1;

%% --- Vorverarbeitung: Datenaufnahme
% --- Daten laden
path2file = "../data/";
filename = "denoising_dataset_File1";
addon = "_Sorted";

load(strcat(path2file, filename, '.mat'));

% --- Bestimmung der Frame-Positionen und Anzahl der Cluster
NoClusterOld = length(unique(frames_cluster));
input_cluster_num = zeros(1, NoClusterOld);
input_cluster_xpos = {};

IteNo = 1;
for idx = min(frames_cluster):1:max(frames_cluster)
    % --- Meta-Informationen einfügen
    input_cluster_xpos{IteNo} = find(frames_cluster == idx);
    input_cluster_num(IteNo) = size(input_cluster_xpos{IteNo},1);
    IteNo = IteNo +1;
end

%% --- Algorithmus zum Cluster-Mergen
new_frames_cluster = int8(zeros(size(frames_cluster)));
new_frames_pos = {};
missed_frames_pos = {};

SelCluster = 0;
IteMiss = 1;
for idx = 1:1:NoClusterOld
    Xin = input_cluster_xpos{idx};
    Yin = double(frames_in(Xin,:));
    metricMean = [];
    mode = 1;

    if(idx == 1)
        %--- Initialisierung
        Xout = Xin;
        SelCluster = 1;    
    else
        %--- Eingangscluster mit bereits gemergten Clustern vergleichen
        for idy = 1:1:size(new_frames_pos, 2)
            % --- Prüfung der Mean-Formen
            Xcheck = new_frames_pos{idy};
            Ycheck = double(frames_in(Xcheck,:));
            Ymean = mean(Ycheck);
            
            WaveRef = crossval(Ymean, Ymean);
            WaveIn = crossval(mean(Yin), Ymean);     

            metricMean(idy, :) = calc_metric(WaveIn, WaveRef);
        end

        
        [candY, candX] = max(metricMean(:, 5));
        
        if(isempty(candX))
            % Keine Lösung vorhanden --> Anhängen
            mode = 1;
        end
        if(candY >= 0.9)
            mode = 2;   
        end

        % --- Ausführung
        switch(mode)
            case 0
                % --- Sammeln und am Ende prüfen
                SelCluster = 0;
                Xout = Xin;
            case 1
                SelCluster = max(new_frames_cluster) +1;
                Xout = Xin;
            case 2
                % --- Bedingung erfüllt    
                SelCluster = candX;
                Xout = [new_frames_pos{SelCluster}; Xin];
    
                % Zweite Prüfung
                Ycheck0 = double(frames_in(Xin, :));
                Ymean0 = mean(Ycheck0);
                Ycheck1 = double(frames_in(Xout,:));
                Ymean1 = mean(Ycheck1);
    
                metricIn = [];
                for idz = 1:1:length(Xin)
                    WaveRef = crossval(Ymean0, Ymean0);
                    WaveIn = crossval(Ycheck0(idz,:), Ymean0);

                    metricIn(idz, :) = calc_metric(WaveIn, WaveRef);
                end
                metricOut = [];
                for idz = 1:1:length(Xout)
                    WaveRef = crossval(Ymean0, Ymean0);
                    WaveIn = crossval(Ycheck1(idz,:), Ymean0);

                    metricOut(idz, :) = calc_metric(WaveIn, WaveRef);
                end

                check.input_mean = mean(metricIn(:,5));
                check.input_std = std(metricIn(:,5));
                check.output_mean = mean(metricOut(:,5));
                check.output_std = std(metricOut(:,5));
                check.diff_mean = check.input_mean - check.output_mean;
                check.diff_std = check.input_std - check.output_std;

                if((abs(check.diff_mean) <= 0.08) && abs(check.diff_std) <= 0.02)
                    %--- Gute Übereinstimmung
                    Xout = [new_frames_pos{SelCluster}; Xin];
                    SelCluster = unique(candX);
                    mode = 3;
                else
                    %--- Schlechte Übereinstimmung
                    Xout = Xin;
                    SelCluster = max(new_frames_cluster) +1;   
                    mode = 4;
                end  
        end   
    end
    
    % --- Übergabe
    new_frames_cluster(Xout) = SelCluster;
    if(SelCluster == 0)
        missed_frames_pos{IteMiss} = Xout;
        IteMiss = IteMiss +1;
    else
        new_frames_pos{SelCluster} = Xout;
    end
end
clear idx idy idz IteNo Xcheck Ycheck;
clear check Ycheck0 Ycheck1 Ymean0 Ymean1 WaveRef WaveIn;

%% --- Nachbearbeitung: Manuelles Cluster mergen
do_manual = 0;
if(do_manual)
    new_frames_pos_change = new_frames_pos;
    new_frames_pos_change{2} = [new_frames_pos{2}; new_frames_pos{6}];
    new_frames_pos_change{6} = [];
    
    e = cellfun('isempty', new_frames_pos_change);
    new_frames_pos_change(e) = [];
    
    new_frames_pos = new_frames_pos_change;
    
    for idx = 1:size(new_frames_pos,2)
        e = new_frames_pos{idx};
        new_frames_cluster(e) = idx;
    end
    clear e idx new_frames_pos_change;
end


%% --- Nachbearbeitung: Unsortierte Frames
N = length(unique(new_frames_cluster));
output_cluster_num = zeros(2, N+1);
output_cluster_num(2,1) = size(missed_frames_pos,2);

for idx = 1:N
    output_cluster_num(1, idx+1) = idx;
    output_cluster_num(2, idx+1) = length(find(new_frames_cluster == idx));
end

if output_cluster_num(2, 1) > 0 
    missed_pos = [];
    for idx = 1:1:size(missed_frames_pos, 2)
        if idx == 1
            missed_pos = missed_frames_pos{idx};
        else 
            missed_pos = [missed_pos; missed_frames_pos{idx}];
        end
    end
    
    unsortedFrames = double(frames_in(missed_pos,:));

    figure(1);
    plot(unsortedFrames', 'b');
    grid on;

    waitforbuttonpress
end

%% --- Nachbearbeitung: Berechnung SNR
calculated_snr = {};
mean_snr = [];
std_snr = [];

for idx = 1:size(new_frames_pos,2)
    Xin = new_frames_pos{idx};
    Yin = double(frames_in(Xin,:));
    Ymean = mean(Yin);

    val_snr = zeros(1, length(Xin));
    for idy = 1:length(Xin)
        val_snr(idy) = calculate_snr(Yin(idy,:), Ymean);
    end
    calculated_snr{idx} = val_snr;
    mean_snr(idx) = mean(val_snr);
    std_snr(idx) = std(val_snr);
end


%% --- Plot results
close all;

if(do_plot)
    NoClusterNew = size(new_frames_pos, 2);
    if(NoClusterNew >= 9)
            SubFig = [3, 4];
    else
        if(NoClusterNew <= 4)
            SubFig = [2, 2];
        else
            SubFig = [3, 3];
        end
    end
    noSubFig = SubFig(1)* SubFig(2);
    sizeID = length(unique(new_frames_cluster));

    for idy = 1:1:ceil(sizeID/noSubFig)
        figure('doublebuffer', 'off', 'visible', 'off');
        set(gcf, 'units','normalized','outerposition',[0 0 0.75 1]);
            
        selID = (1:1:noSubFig) + noSubFig*(idy-1);
        if(selID(end) > sizeID)
            selID = selID(1):1:sizeID;
        end
    
        IteNo = 1;
        for idx = 1:1:length(selID)
            Yin = frames_in(new_frames_pos{selID(idx)},:);
            subplot(SubFig(1), SubFig(2), IteNo);
            IteNo = IteNo +1;
            
            plot(Yin', 'b');
            hold on; grid on;
            plot(mean(Yin), 'r', 'Linewidth', 1);
    
            title(strcat("Cluster ID: ",num2str(selID(idx)), " - SNR = ", num2str(mean_snr(idx)), "(\pm", num2str(std_snr(idx)), ") dB - No = ", num2str(size(Yin,1))));
            set(gca, 'FontSize', 14);
        end        
        saveas(gcf, strcat('Bilder/', filename, '_Fig', num2str(idy, '%02d'), '.jpg'));
    end

    clear idx idy;
end


%% --- Transfer to new file
if(do_save)
    frames_in = frames_in;
    frames_cluster = int16(new_frames_cluster);

    save(strcat(path2file, filename, addon, ".mat"), 'frames_in', 'frames_cluster');

    clear FileName;
end

%% --- Ende
X = 1:1:100;
Y = shuffle(X);

disp("Ende");

%% --- External function
function out = calc_metric(WaveIn, WaveRef)
    [Y1, X1] = max(WaveIn);
    [Y2, X2] = max(WaveRef);
    
    out = [];
    % --- Normal results
    out(1) = X1 - X2;
    out(2) = sum(abs(WaveIn - WaveRef).^2);
    out(3) = trapz(WaveIn(1:X1)) - trapz(WaveIn(X1:end));
    out(4) = X1;
    out(5) = Y1;
end

function out = shuffle(in)
     out = in(randperm(length(in)));
 end

function out = crossval(Wave1, Wave2)
    out = xcorr(Wave1, Wave2)/(sqrt(sum(Wave1.^2))* sqrt(sum(Wave2.^2)));
end

function outdB = calculate_snr(waveIn, waveOut)    
    A = sum(waveIn.^2);
    B = sum((waveOut - waveIn).^2);
    outdB = 10*log10(A/B); 
end

function checkData(WaveIn, noID)
    WaveRef = mean(WaveIn);
    Yref = crossval(WaveRef, WaveRef);
    metric_test = zeros(size(WaveIn, 1), 5);
           
    for idz = 1:1:size(WaveIn, 1)
        Yin = crossval(WaveIn(idz,:), WaveRef);
        metric_test(idz, :) = calc_metric(Yin, Yref);
    end
    
    str = strcat('Cluster-ID: ', num2str(noID), ' - Crossval, Mean = ', num2str(mean(metric_test(:,5))), ' +/- ', num2str(std(metric_test(:,5))));
    disp(str);

    figure();
    subplot(2,1,1);
    p1 = plot(WaveIn', 'b', 'Linewidth', 1);       
    hold on;                grid on;
    p2 = plot(WaveRef, 'r', 'Linewidth', 1);  
    title(str);

    subplot(2,1,2);
    p3 = histogram(metric_test(:,5),100);

    % --- Warte-Routine
    waitforbuttonpress

    close gcf;    
end