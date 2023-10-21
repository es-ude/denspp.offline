close all;
clear all;
clc;

load('2023-05-15_Dataset01_SimDaten_Martinez2009_Sorted.mat')

cluster_id = unique(frames_cluster);
cluster_no = size(cluster_id, 1);

frames_sel = {};
frames_mean = {};

for idx = 1:1:cluster_no
    xsel = find(frames_cluster == cluster_id(idx));
    frames_sel{idx} = frames_in(xsel,:);
    frames_mean{idx} = mean(frames_in(xsel,:));
end

%% --- Plotten
figure(1);
color = {'k'; 'r'; 'b'; 'g'; 'm'};
for idx = 1:1:cluster_no
    sel_plot = 4*(idx-1);
    subplot(3, cluster_no, idx);
    plot(frames_sel{idx}', 'Color',  [0.5 0.5 0.5]);
    hold on;
    plot(frames_mean{idx}, color{idx});

    subplot(3, cluster_no, idx + 5);
    plot(diff(frames_sel{idx}'), frames_sel{idx}(:,2:end)', 'Color',  [0.5 0.5 0.5]);
    hold on;
    plot(diff(frames_mean{idx}), frames_mean{idx}(2:end), color{idx});

    subplot(3, cluster_no, idx + 10);
    plot(diff(diff(frames_sel{idx}')), frames_sel{idx}(:,3:end)', 'Color',  [0.5 0.5 0.5]);
    hold on;
    plot(diff(diff(frames_mean{idx})), frames_mean{idx}(3:end), color{idx});
end

