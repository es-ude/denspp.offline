close all;  clear all;  clc;
line = 1.5;

q = 1.602e-19;
kB = 1.380649e-23;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% --- Header
    %Author:            A. Erbslöh
    %Erstelldatum:      09.06.2020
    %Aktualisierung:    20.07.2020
    %Company:           EBS
    %Version:           1v0
    %Projekt:           Modellierung von neuronalen Aktivitäten
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Modellierung der AP-Propagation innerhalb eines Axons');
disp('-----------------------------------------------------');

%% --- Parameter
saveGIF = 0;
% --- Stimulationsparameter
StimAmp = 16e-6;        % Stim.-Amplitude [A o. V] for VCS, CCS
NoisePeak = 0;          % Rauschamplitude [A]
Tph = 0.100e-3;         % Phasendauer [s]
Tipd = 0.02e-3;         % Interphasenverzögerung [s]
noWFG = 1;              % Waveform pattern (1-6)
firstWFG = 0;           % 0: cathodic, 1: anodic
noWFG1 = 1;             % 0: monophasisch, 1: biphasisch
noStim = 2;             % 1: VCS, 2: CCS, 3: ChCS

% --- Elektroden-Eigenschaften
Cdl = 4e-9;             % Kapazität der Doppelschicht 
Rfar = 1e6;             % Faraday-Widerstand der Doppelschicht
Cmod = 10.8e-12;          % ChCS-Modulatorkapazität
Qmod = 23e-12;
fmod = 0.01e6;           % ChCS-Modulatorfrequenz

% --- Position der nähesten Nodes von Elektrode aus [x y]
xyElec = 1e-6* [0 0];
xyNode = 1e-6* [28 0.5];
% --- Gewebe-Eigenschaften
noNode = 61;
dAxon = 2e-6;
dMym = 3e-6;
LAxon = 50e-6;
LNode = 10e-6;
rhoMedium = 100e-2;
rhoAxon = 130e-2;
Cmem = 4e-2;

% --- Modellparameter
nNode = 6;              % Anzahl an Nodes zum Plotten
T = 300.18;             % Körpertemperatur
dt0 = 1e-6;           % Temporal resolution or time step (without stimulation) [s]
dt1 = 50e-9;           % Temporal resolution or time step (during stimulation) [s]
dT0 = 1e-3;             % Zeitpunkt der Stimulation [s]
TSim = 5e-3;           % maximale Simulations-Dauer

%% --- Vorbereitung des McNeal-Modells
noMitte = 1 + floor(noNode/2);
Node = noMitte + floor(linspace(0, noMitte-1, nNode));
% --- Stabilitätsfaktor
rAxon = rhoAxon* LAxon;
Raxon = 4* rhoAxon* LAxon/(pi* dAxon^2);
tau = Raxon* Cmem* pi* dAxon* LAxon;
ath = LAxon^2/tau;
CFL = ath* dt0/LAxon.^2; 

% --- Vorbereitung
switch(noStim)
    case 1
        disp('Spannungsgesteuerte Stimulation , VCS');
        dt1 = dt0;
    case 2
        disp('Stromgesteuerte Stimulation, CCS');
        dt1 = dt0;
    case 3
        disp('Ladungsgesteuerte Stimulation, ChCS');
        dt1 = dt1;
end

% --- Medium-Widerstand bestimmen
Rmed = zeros(1, noNode);
for i = 1:1:noNode
    Rmed(i) = rhoMedium/(2*pi*norm(xyNode + [0 (i-noMitte)*LAxon] - xyElec));
end
Rtis = max(Rmed);

cd 2_Funktionen
% --- Erstellen der Strom-Anregung
Tstim = Tph + noWFG1*(Tph + Tipd); 
Tmax = max([TSim (2*dT0 + Tstim)]);
t = 0:dt0:Tmax;
I_EL = zeros(size(t)); 

% --- Erstellen der WFG
I_EL0 = zeros(size(t)); 
T0 = t;
dT = 0;
% Einsetzen der Stimulationsarray
for i = 1:1:noWFG1+1
    I0 = StimAmp* (-1)^(firstWFG+i);
    A = find(t >= dT0 + dT);
    B = find(t >= dT0 + dT + Tph);
    Istim0 = waveformGenerator(noWFG, I0, int16((B(1)-A(1)+1)*dt0/dt1));
    Istim1 = stimModulation(noStim, dt1, Istim0, [Rtis Cdl], [Cmod fmod Qmod 200e-9]);
    I_EL0 = [I_EL(1:A(1)-1), Istim1, I_EL(B(1)+1:end)];
    I_EL = I_EL0;
    T1 = t(A(1)) + (t(B(1))-t(A(1)))/(length(Istim1)-1)*(0:1:length(Istim1)-1);
    T0 = [t(1:A(1)-1), T1, t(B(1)+1:end)];
    t = T0;
    dT = Tph + Tipd;
end
I_EL = I_EL + NoisePeak*(rand(size(t))-0.5);
clear A B Istim0 Istim1 T1 T0;

% --- Arrays deklarieren
U_WE = zeros(size(t));
Q_dl = zeros(size(t));
U_dl = zeros(size(t));

Qinj = zeros(noNode, length(t));
Einj = zeros(noNode, length(t));
Vint = zeros(noNode, length(t));
Vext = zeros(noNode, length(t));
Vneu = zeros(noNode, length(t));
Ineu = zeros(noNode, length(t));

dV = zeros(noNode, length(t));
m = zeros(noNode, length(t)); 
n = zeros(noNode, length(t)); 
h = zeros(noNode, length(t));

%% --- Computing Neural Network (McNeal-Modell)
disp(['Simulationsdauer: ', num2str(1e3* t(end)), ' ms']);
pbar = ProgressBar(length(t)-1, 'Execution Status: ', 0);
dt = [diff(t), dt0];
for i = 1:1:length(t)-1
    % --- Verhalten des Elektroden-Interfaces
    U_dl(i+1) = U_dl(i)*(1 - dt(i)/(Rfar* Cdl)) + I_EL(i)/Cdl*dt(i);
    Q_dl(i+1) = Cdl* U_dl(i);
    U_WE(i+1) = Rtis* I_EL(i) + U_dl(i);
    
    % --- Berechnung des extrazellulären Potentials am Node
    Vext(:, i) = Rmed.* I_EL(i); 

    % --- Berechnung der Ladungsmenge und Energie
    if(i > 1)
        Qinj(:, i) = Qinj(:, i-1) + Vext(:, i-1)/Raxon* dt(i);
        Einj(:, i) = Einj(:, i-1) + Rtis* (Vext(:, i-1)/Raxon).^2* dt(i);
    end
    
    % --- Neuronen-Berechnung (mit Crank-Nichelson-Verfahren)
    m0 = m(:, i);
    n0 = n(:, i);
    h0 = h(:, i);
    %[Vint1, m1, n1, h1] = modellMcNeal_0v0(Vext(:, i), Vint(:, i), rAxon, T, 1e3*dt, i, m0', n0', h0');
    [Vint1, m1, n1, h1] = modellMcNeal_0v1(Vext(:, i), Vint(:, i), T, dt(i), i, m0', n0', h0', CFL*dt(i)/dt0);
    m(:, i+1) = m1';
    n(:, i+1) = n1';
    h(:, i+1) = h1';
    Vint(:, i+1) = 1e-3*Vint1';
    
    pbar.update(i);
end  
pbar.finish();

clear m0 m1 n0 n1 h0 h1 Vint1 pbar kT i;

%% --- Nachbearbeitung
% numerische Stabilität prüfen
G = 0;
 for i = 2:1:length(t)
    G(i-1) = mean(Vint(:,i)./Vint(:,i-1)); 
 end

Vint(:,1) = Vint(:,2);  
Einj(:,end) = Einj(:,end-1);
Qinj(:,end) = Qinj(:,end-1);
Einj(:,end) = Einj(:,end-1);

Q0 = [min(Q_dl) max(Q_dl)];
Q1 = [min(Qinj(noMitte, :)) max(Qinj(noMitte, :))];
E0 = max(Einj(noMitte,:));

% --- Berechnung der Feuerrate
VmaxAP = {};
xAP = {};
for i = 1:1:length(Node)
    [VmaxAP{i}, xAP{i}] = findpeaks(Vint(Node(i),:), 'MinPeakHeight', 0, 'MinPeakDistance', 1.5e-3/dt0);
    activeAP(i) = isempty(xAP{i});
end

% --- Neuronen-Verhalten gemessen an der Elektrode
R_L = 100*Rtis;
tau0 = (Rmed' + R_L)/Raxon* tau;
V0 = zeros(size(t));
for i = 2:1:length(t)
    Ineu(:, i) = Ineu(:, i-1).*(1 - dt(i)./tau0) + (Vint(:,i) - Vint(:, i-1)).* Rmed'./(Rmed' + R_L);
    Vneu(:, i) = R_L* Ineu(:, i);          
end   
% for i = 1:1:noNode
%     V0 = V0 + Vneu(i,:);
% end
Vn = sqrt(4*kB*T*Rtis/dt0)* 2.*(rand(size(Vint))-0.5);
Vneu = Vneu + 1e6*Vn;

%% --- Ausgabe
disp('-----------------------------------------------------');
disp('Auswertung');
disp(['CFL-Faktor: '            num2str(CFL)]);         
disp(['Axon-Widerstand: ',      num2str(1e-3*Raxon),        ' kOhm']);
disp(['Medium-Widerstand: ',    num2str(1e-3*max(Rmed)),    ' kOhm']);
disp(['Zeitkonstante: ',        num2str(1e6* tau),          ' µs']);
disp(['Propagation: ',          num2str(LAxon/tau),         ' m/s']);
disp(['Inj. Ladungsmenge: ',    num2str(1e9*Q0(firstWFG+1)),' nC']);
disp(['Inj. Q_Neuron: ',        num2str(1e12*Q1(firstWFG+1)),' pC']);
disp(['Energiegehalt: ',        num2str(1e12*E0),           ' pJ']);
disp(['Stim. erfolgreich? ',    num2str(1-min(activeAP))]);
disp(['Erregungsleitung? ',     num2str(1-activeAP(end))]);
if(activeAP(1) == 0)
    disp(['Post-Stim.-Zeit: ',  num2str(1e3*mean(t(xAP{1})-dT0)), ' ms']);
end

clear Q0 Q1 E0;

%% --- Plotten - Figure 1
close all;
ColorLine = 'krbymrb';

figure(1);
subplot(3,1,1);
yyaxis left;    plot(1e3* t, 1e6* I_EL, 'k', 'Linewidth', line);    ylabel('I_{EL} / µA');
yyaxis right;   plot(1e3* t, U_WE, 'r', 'Linewidth', line);         ylabel('U_{CE} / V');
grid on;

setGraphicStyle(0);
legend({'I_{EL}', 'U_{CE}'}, 'Location', 'NorthEast');

% Subplot
subplot(3,1,2); hold on;

for i = 1:1:length(Node)
    plot(1e3* t, 1e3* Vint(Node(i), :), ColorLine(i), 'Linewidth', line);
end
grid on; 

setGraphicStyle(0);
ylabel('U_{int} / mV');

% Subplot
subplot(3,1,3); hold on;
for i = 1:1:length(Node)
    plot(1e3* t, Vneu(Node(i), :), ColorLine(i), 'Linewidth', line);
end
grid on;

setGraphicStyle(1);
ylabel('U_{ext} / µV');
xlabel('Zeit t / ms');

% --- Plotten - Figure 2
figure(2);
surface(1e3*t, 1:1:noNode, Vint, 'LineStyle', 'none');
grid on;
colorbar();

setGraphicStyle(1);
xlabel('Zeit t / ms');
ylabel('Node');
zlabel('Spannung U_{int} / V');

% --- Plotten - Stabilität
% figure(3);
% plot(G);

%% --- Figure 3
if(saveGIF)
    noSamples = 60;
    dT = floor(length(t)/noSamples);
    filename = 'AP_Propagation.gif';

    figure(4);
    for i = 0:1:noSamples
        plot(1:noNode, 1e3* Vint(:, 1 + i* dT), 'k', 'Linewidth', line);
        grid on;
        setGraphicStyle(0);
        xlabel('Node');
        ylabel('U_{int} / mV');
        ylim([-80 40]);
        xlim([1 noNode-1]);
        title(['t = ', num2str(1e3*t(1 + i* dT)), ' ms']); 
        pause(0.5);

        % GIF erstellen
        frame = getframe(3);
        im = frame2im(frame);
        [imind,cm] = rgb2ind(im,256);
        if i == 0
            imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
        else
            imwrite(imind,cm,filename,'gif','WriteMode','append');
        end
    end
end