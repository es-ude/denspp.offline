close all;
clear all;
clc;

line = 1;
woChBLC = 2;

%% --- Berechnung
switch(woChBLC)
    case 1
        Data = csvread('200707_ChBLCwo.csv');
    case 2
        Data = csvread('200707_ChBLC.csv');
end
% --- Sim. Daten
t = 1e3* Data(:,1);
Qinj = Data(:,2);
dUel = Data(:,3);

% --- abstrakte Größen
switch(woChBLC)
    case 1
        Data = csvread('200707_ChBLCwo_abstract.csv');
    case 2
        Data = csvread('200707_ChBLC_abstract.csv');
end
ta = Data(2:end,1);
dQ1 = Data(2:end,2);
dQ2 = Data(2:end,4);
nAno = Data(2:end,3);

t0 = 0.38;
dT = 0.5;
i = 0;
dQ = 0;
nA = 0;
Udl = 0;
while(t0 <= t(end))
    x0 = find(t0 <= t);
    Udl(i+1) = dUel(x0(1));
    
    i = i+1;
    t0 = t0 + dT;
end

%% --- Plotten
close all;

% --- Plot 1
figure(1);
subplot(4,1,1);
plot(t, dUel, 'k', 'Linewidth', line);
ylabel('\Delta U_{EL} / V');
setGraphicStyle(0);
ylim([-1.5 1.5]);
yticks(-1.5:0.5:1.5);
xticks(0:4:20);

subplot(4,1,2);
plot(t, Qinj, 'k', 'Linewidth', line);
ylabel('Q_{inj} / nC');
setGraphicStyle(0);
ylim([-1.5 0.1]);
yticks(-1.5:0.3:0);
xticks(0:4:20);

subplot(4,1,3);
yyaxis left;
plot(1e3*ta, dQ2, 'k', 'Linewidth', 0.5, ...
    'Marker', '.', 'MarkerSize', 10);
ylabel('\Delta Q / pC');
xticks(0:4:20);

yyaxis right;
plot(1e3*ta, nAno, 'Color', [0.502 0.502 0.502], 'Linewidth', 0.5, ...
    'Marker', '.', 'MarkerSize', 10);
ylabel('n_{ano}');
legend({'\Delta Q', 'n_{ano}'}, 'Location', 'SouthWest');
setGraphicStyle(0);
xticks(0:4:20);

subplot(4,1,4);
plot(1e3*ta, 1e3* Udl, 'k', 'Linewidth', 0.5, ...
    'Marker', '.', 'MarkerSize', 10);
ylabel('U_{dl} / mV');
xlabel('Zeit t / ms');
setGraphicStyle(1);
xticks(0:4:20);