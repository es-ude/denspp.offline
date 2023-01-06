close all;  clear all;  clc;

load("denoising_dataset.mat")

%% --- Bestimmung der Frame-Positionen und Anzahl der Cluster
Cluster_NumCnt = zeros(1, max(frames_cluster));
Cluster_Xpos = {};

iteration = 1;
for idx = min(frames_cluster):1:max(frames_cluster)
    Cluster_Xpos{iteration} = find(frames_cluster == idx);
    Cluster_NumCnt(iteration) = size(Cluster_Xpos{iteration},1);
    iteration = iteration +1;
end
clear idx iteration;

%% --- Korrelation zur Bestimmung der gemeinsamen Frames
metric = [];
frames_mean1 = [];
for idy = 1:length(Cluster_NumCnt)
    Xin = Cluster_Xpos{idy};
    
    % --- Mittelwert bestimmen
    frames_mean0 = mean(Frames_in(Xin,:));
    peakVal = abs([min(frames_mean0), max(frames_mean0)]);
    if(peakVal(1) <= peakVal(2))
        normVal = peakVal(2);
    else
        normVal = peakVal(1);
    end
    frames_mean0 = frames_mean0/normVal;
    
    % --- Kreuzkorrelation
    metric0 =  [];
    for idx = 1:1:length(Xin)
        corVal_id = xcorr(frames_mean0, frames_mean0);
        corVal_no = xcorr(frames_mean0, Frames_in(Xin(idx),:)/normVal);
        
        if(idx == 1)
            metric0 = calc_metric(corVal_no, corVal_id);
        else
            metric0 = [metric0; calc_metric(corVal_no, corVal_id)];
        end
    end

    % --- Backprocessing
    if(idy == 1)
        metric = [mean(metric0), std(metric0)];
        frames_mean1 = frames_mean0;
    else 
        metric = [metric; mean(metric0), std(metric0)];
        frames_mean1 = [frames_mean1; frames_mean0];
    end
end

%% --- Korrelation zwischen Clustern
Xref = 2;

X0 = frames_mean1(Xref,:);
X1 = frames_mean1(Xref,:);
Y0 = 1;

h = figure(1);
subplot(2,1,1);
p1 = plot(X0, 'k', 'Linewidth', 1);  
grid on;    hold on;
p2 = plot(X1, 'r', 'Linewidth', 1);

subplot(2,1,2);
p3 = plot(Y0, 'k', 'Linewidth', 1);  
grid on;

title(strcat("Index: 1, Ref.:", num2str(Xref)));

p1.YDataSource = "X0";
p2.YDataSource = "X1";
p3.YDataSource = "Y0";

for idy = 1:1:length(Cluster_NumCnt)
    if(idy ~= Xref)
        Y0 = xcorr(frames_mean1(idy,:), frames_mean1(1,:));
        X1 = frames_mean1(idy,:);
        X0 = frames_mean1(Xref,:);
        title(strcat("Index: ", num2str(idy), ", Ref.: ", num2str(Xref)));
       
        refreshdata;
        drawnow;
        pause(0.1);
        
        saveas(gcf, strcat('Bilder/Idx',num2str(idy),'_Ref',num2str(Xref),'.jpg'))
    end
end


%% --- Plotten
figure(2);
tiledlayout(2,1)

% --- Normalframe
nexttile();
plot(Frames_in(Xin,:)'/normVal, 'b', 'Linewidth', 1);
hold on;    grid on;
plot(frames_mean0, 'r', 'Linewidth', 2);

% --- Kreuzkorrelation
nexttile();
plot(corVal_no, 'b', 'Linewidth', 1);
hold on;    grid on;
plot(corVal_id, 'r', 'Linewidth', 2);


%% --- External function
function out = calc_metric(Wave1, Wave2)
    X1 = find(Wave1 == max(Wave1),1);
    X2 = find(Wave2 == max(Wave2),1);
    
    out = [];
    out(1) = max(Wave1) - max(Wave2);
    out(2) = find(Wave2 == max(Wave2),1) - find(Wave1 == max(Wave1),1);
    out(3) = sum(abs(Wave1 - Wave2).^2);
    out(4) = trapz(Wave1);
    out(5) = trapz(Wave1(1:X1-1)) - trapz(Wave1(X1:end));
    out(6) = trapz(Wave2);
end