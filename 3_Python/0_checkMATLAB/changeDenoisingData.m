close all;  clear all;  clc;

%% --- Settings
% 0: all waveforms - 1: all mean_waveforms
mode = 1;

do_plot = 0;
do_norm = 0;
do_save = 1;

%% --- Vorverarbeitung: Datenaufnahme
% --- Daten laden
load("../data/denoising_dataset.mat")

% --- Bestimmung der Frame-Positionen und Anzahl der Cluster
Cluster_NumCnt = zeros(1, max(frames_cluster));
Cluster_Xpos = {};
Cluster_ID = [];

IteNo = 1;
for idx = min(frames_cluster):1:max(frames_cluster)
    Cluster_Xpos{IteNo} = find(frames_cluster == idx);
    Cluster_NumCnt(IteNo) = size(Cluster_Xpos{IteNo},1);
    Cluster_ID(IteNo) = idx;
    IteNo = IteNo +1;
end

% --- Mittelwert bestimmen
frames_mean = zeros(length(Cluster_NumCnt), size(frames_in,2));

for idx = 1:length(Cluster_NumCnt)
    Yin = Cluster_Xpos{idx};
    
    % --- Mittelwert bestimmen
    Y0 = mean(frames_in(Yin,:));
    peakVal = abs([min(Y0), max(Y0)]);
    if(peakVal(1) <= peakVal(2))
        normVal = peakVal(2);
    else
        normVal = peakVal(1);
    end
    
    if (do_norm)
        frames_mean(idx,:) = (Y0 / normVal);
    else
        frames_mean(idx,:) = Y0;
    end
end
clear normVal idx Y0 IteNo;

%% --- Vorverarbeitung: Korrelation und Plot
% --- Mode distinguiation
if mode == 0
    % --- All Waveforms
    Yin = frames_in;
    XrefTest = find(frames_cluster == 2, 1);
    IteMin = 1000; 
    XPos = {};
else
    % --- All mean waveforms
    Yin = frames_mean;
    XrefTest = 2;
    IteMin = 10; 
    XPos = Cluster_Xpos;
end

% --- Preparing plot
if(do_plot)
    X0 = Yin(Xref,:);  Y0 = 1;
    X1 = Yin(Xref,:);  Y1 = 1;
    
    figure('visible', 'off');
    subplot(2,1,1);
    p1 = plot(X0, 'k', 'Linewidth', 1);  
    grid on;    hold on;
    p2 = plot(X1, 'r', 'Linewidth', 1);
    
    xlim([1 size(Yin,2)]);
    t0 = title(strcat("Index: ", num2str(1), " Ref.:", num2str(Xref)));
    
    subplot(2,1,2); 
    p3 = plot(Y0, 'k', 'Linewidth', 1);  
    grid on;    hold on;
    p4 = plot(Y1, 'r', 'Linewidth', 1);
    legend({'Reference'; 'Input'}, 'Location', 'northeast');
    
    ylim([-1 1]);   xlim([1 2*size(Yin,2)-1]);
    t1 = title(strcat('R2 = ', num2str(0), ' - Sym =', num2str(0),' - Ymax = ', num2str(0)));
    
    p1.YDataSource = "X0";
    p2.YDataSource = "X1";
    p3.YDataSource = "Y0";
    p4.YDataSource = "Y1";
end

%% --- Ausführung
frames_meanNew = [];
frames_ID = zeros(size(frames_in,1),1);

IteNo = 1;
checkLoop = 1;
while(checkLoop)
    Xref = 1;
    NoRept = size(Yin,1);
    % --- Do metric calculation
    metric0 = zeros(NoRept, 6);
    Y0 = crossval(Yin(Xref,:), Yin(Xref,:));
    
    for idx = 1:1:size(Yin,1)
        Y1 = crossval(Yin(idx,:), Yin(Xref,:));
        X1 = Yin(idx,:);
        X0 = Yin(Xref,:);        
        metric0(idx, :) = [idx, calc_metric(Y1, Y0)];
    
        % Plotting
        if do_plot
            t0.String = strcat("Index: ", num2str(idx), " - Ref.:", num2str(Xref));
            t1.String = strcat('R2 = ', num2str(metric0(idx,3)), ' - Ymax = ', num2str(metric0(idx,6)));
    
            refreshdata;
            drawnow;
            pause(0.1);
            saveas(gcf, strcat('Bilder/Xcorr_Ref',num2str(Xref), '_Idx', num2str(idx), '.jpg'));
        end
    end
    %% --- Table generieren
    metric_idx = metric0(:,1);
    metric_Delay = metric0(:,2);
    metric_R2 = metric0(:,3);
    metric_Sym = metric0(:,4);
    metric_Xmax = metric0(:,5);
    metric_Ymax = metric0(:,6);
        
    metric = table(metric_idx, metric_Delay, metric_R2, metric_Sym, metric_Xmax, metric_Ymax);
    metric = sortrows(metric, 'metric_Ymax', 'descend');
    
    clear metric0 idx;

    if(IteNo == 1 && mode == 1)
        Cluster_MetaInfo = table(Cluster_Xpos', Cluster_NumCnt', Cluster_ID');
    end
    
    % --- Cluster mergen
    X00 = find((metric.metric_R2 <= 0.12) & (metric.metric_Ymax >= 0.95));
    X01 = metric.metric_idx(X00);

    Pos = [];
    % Position of ClusterID
    for idx = 1:1:length(X01)
        if(idx == 1)
            Pos = cell2mat(table2array(Cluster_MetaInfo(idx,1)));
        else
            Pos = [Pos; cell2mat(table2array(Cluster_MetaInfo(idx,1)))];
        end
    end
    
    % Mean waveform
    if(length(X01) == 1)
        mean_frame = Yin(X01,:);
    else
        mean_frame = mean(Yin(X01,:));
    end 


    % --- Deleting
    Yin(X01,:) = [];
    metric(X00,:) = [];
    Cluster_MetaInfo(X01,:) = [];

    % --- End control
    frames_ID(Pos) = IteNo-1;
    if(IteNo == 1)
        frames_meanNew = mean_frame;
    else
        frames_meanNew = [frames_meanNew; mean_frame];
    end
    IteNo = IteNo +1;

    % --- Check Loop condition
    if(isempty(Yin))
        checkLoop = 0;
    else
        checkLoop = (IteNo <= IteMin) || (metric.metric_Ymax(1) >= 0.8);
    end    
end
clear idx X0 X1 Y1 p1 p2 p3 p4 t0 t1 X00 X01;

%% --- Plotten
close all;

% --- Alle mean-Waveforms
figure(3);
subplot(2,1,1);
plot(frames_mean', 'Linewidth', 1);
grid on;

subplot(2,1,2);
plot(frames_meanNew', 'Linewidth', 1);
grid on;

% --- Histogramm (vorher/nachher)
figure(4);
subplot(2,1,1);
histogram(frames_cluster(1:end), length(unique(frames_cluster(1:end))));

subplot(2,1,2);
histogram(frames_ID(2:end), length(unique(frames_ID(2:end))));

% --- Alle Cluster einzeln mit mean Waveform
sizeID = length(unique(frames_ID));
for idy = 1:1:ceil(sizeID/9)
    figure('doublebuffer', 'off', 'visible', 'off');
    set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
        
    selID = (1:1:9) + 9*(idy-1);
    if(selID(end) > sizeID)
        selID = selID(1):1:sizeID;
    end

    for idx = 1:1:length(selID)
        subplot(3,3,idx);
        
        A = find(frames_ID == selID(idx)-1);
        plot(frames_in(A,:)', 'b');
        hold on; grid on;
        plot(frames_meanNew(selID(idx),:), 'r', 'Linewidth', 1);
        title(strcat("Cluster ID: ",num2str(selID(idx)-1)));
    end
    saveas(gcf, strcat('Bilder/New_IDset_',num2str(idy), '.jpg'));
    close gcf;
end

%% --- Transfer to new file
if(do_save)
    frames_in = frames_in;
    frames_cluster = int16(frames_ID);
    frames_mean = frames_meanNew;

    FileName = "_v2";
    save(strcat("../data/denoising_dataset", FileName, ".mat"), 'frames_in', 'frames_cluster', 'frames_mean');
end

%% --- Ende
disp("Ende");

%% --- External function
function out = calc_metric(WaveIn, WaveRef)
    [Y1, X1] = max(WaveIn);
    [Y2, X2] = max(WaveRef);
    
    out = [];
    %% --- Normal results
    out(1) = X1 - X2;
    out(2) = sum(abs(WaveIn - WaveRef).^2);
    out(3) = trapz(WaveIn(1:X1)) - trapz(WaveIn(X1:end));
    out(4) = X1;
    out(5) = Y1;
end

function out = crossval(Wave1, Wave2)
    out = xcorr(Wave1, Wave2)/(sqrt(sum(Wave1.^2))* sqrt(sum(Wave2.^2)));
end