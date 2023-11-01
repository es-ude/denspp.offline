clear all;
close all;
clc;

load('SDA_Dataset.mat')

%% ---- Plotting
figure(1);

idx = 21;

subplot(2,1,1);
plot(sda_in(idx, :), 'k', 'Linewidth', 1);

subplot(2,1,2);
plot(sda_pred(idx, :), 'r', 'Linewidth', 1);
