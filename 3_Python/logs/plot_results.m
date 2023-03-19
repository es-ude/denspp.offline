close all;
clear all;
clc;

%file_input = '20230315-000000_dnn_dae_v2';
file_input = '20230316-000000_dnn_dae_v2';
%file_input = '20230316-000000_cnn_dae_v2';

%% --- Pre-Processing
load(file_input)

NoCluster = unique(Cluster);
for idx = 1:1:length(unique(Cluster))
    marker = NoCluster(idx);
    mark{idx} = find(Cluster == marker); 

    mean_frames(idx,:) = mean(YPred(mark{idx},:));
end



%% --- Plotting
color = {'b'; 'r'; 'g'; 'k'};

figure(1)
tiledlayout(1,3)

nexttile;
plot(PredIn', 'LineWidth', 1)
grid on;
title('Network Training');

nexttile;
plot(YPred', 'LineWidth', 1);
grid on;    hold on;
for idx = 1:1:length(unique(Cluster))
    plot(mean_frames(idx,:)', 'LineWidth', 5, 'Color', color{idx});
end
title('Network Predicted');

nexttile;
selFeat = [3 5];
plot(Feat(mark{1}, selFeat(1)), Feat(mark{1}, selFeat(2)), 'LineStyle', 'none', 'Marker', '.', 'MarkerSize', 16, 'Color', color{1});
grid on;    hold on;
for idx = 2:1:length(unique(Cluster))
    plot(Feat(mark{idx}, selFeat(1)), Feat(mark{idx}, selFeat(2)), 'LineStyle', 'none', 'Marker', '.', 'MarkerSize', 16, 'Color', color{idx});
end
title('Feature Map from DAE');