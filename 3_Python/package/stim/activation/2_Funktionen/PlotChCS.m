close all;  clear all;  clc;
line = 1.5;

fig = openfig('1_Bilder/200910_SimNeuron_BiCatChCS_Rect_d12um.fig');
a = get(gca,'Children');

TphS = 1e-3*get(a, 'XData');

Qth = get(a, 'YData');
Qinj_MPC1 = 10.8e-12* 1.65* TphS* 430e3;
Qinj_MPC2 = 4* 3* 2.4* 1.6e-12* TphS* 1e6;

%% --- Plotten
close all;
figure(1);
semilogx(1e3*TphS, 1e9* Qinj_MPC1, 'k', 'Linewidth', line);
hold on;    grid on;
semilogx(1e3*TphS, 1e9* Qinj_MPC2, 'r', 'Linewidth', line);
semilogx(1e3*TphS, Qth, 'b', 'Linewidth', line);

xlabel('Phasendauer T_{ph} / ms');
ylabel('Q_{inj} / nC');
legend({'max(Q_{MPC1})', 'max(Q_{MPC2})', 'Q_{th}'}, 'Location', 'SouthEast');

setGraphicStyle(1);
ylim([0 5]);
yticks(0:0.5:5);
