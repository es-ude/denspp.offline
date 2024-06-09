close all;  clear all;  clc;
warning('off', 'all');
line = 1.5;

q = 1.602e-19;
kB = 1.380649e-23;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% --- Header
    %Author:            A. Erbslöh
    %Erstelldatum:      09.06.2020
    %Aktualisierung:    12.06.2020
    %Company:           EBS
    %Version:           1v0
    %Projekt:           Modellierung von neuronalen Aktivitäten 
    %                   Parameter-Sweep zur Herleitung der
    %                   Stärke-Dauer-Beziehung (Abstand zwischen Elektrode und Gewebe)
    %Hinweis:           Bei ChCS wird Abstand zwischen 10 µm und 1 mm variiert
    %                   Bei CCS wird Abstand zwischen 1 µm und 100 µm variiert
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Stärke-Dauer-Beziehung des 2D-Modells (Abstand)');
disp('-----------------------------------------------------');

%% --- Parameter
% --- Stimulationsparameter
StimStart_I = 1e-6;                 % Stim.-Amplitude [A o. V]
StimStart_f = 50e3;                % ChCS-Modulatorfrequenz
NoisePeak = 0;                      % Rauschamplitude [A]
TphS = 0.2e-3;                      % Phasendauer [s]
Tipd = 0.001e-3;                    % Interphasenverzögerung [s]
patternWFG = 1;                     % Waveform Pattern (1-7) 
phaseWFG = 1;                       % 0: monophasisch, 1: biphasisch
firstStim = 0;                      % 0: cathodic, 1: anodic
ModulationStim = 3;                 % 1: VCS, 2: CCS, 3: ChCS

% --- Elektroden-Eigenschaften
Cdl = 4e-9;             % Kapazität der Doppelschicht 
Rfar = 1e6;             % Faraday-Widerstand der Doppelschicht
Cmod = 20e-12;          % ChCS-Modulatorkapazität
Qmod = 23e-12;

% --- Position der nähesten Nodes von Elektrode aus [x y]
Dis = logspace(0, 2, 101); 
xyElec = 1e-6* [0 0.5];

% --- Gewebe-Eigenschaften
noNode = 31;
dAxon = 2e-6;
dMym = 3e-6;
LAxon = 50e-6;
LNode = 10e-6;
rhoMedium = 100e-2;
rhoAxon = 130e-2;
Cmem = 4e-2;
Cmye = 0.1e-2;

% --- Modellparameter
nNode = 10;                 % Anzahl an Nodes zum Plotten
T = 300.18;                 % Körpertemperatur
dt0 = 1e-6;                 % Temporal resolution or time step (without stimulation) [s]
dt1 = 50e-9;                % Temporal resolution or time step (during stimulation) [s]
dT0 = 1e-3;                 % Zeitpunkt der Stimulation [s]
TSim = 12e-3;               % maximale Simulations-Dauer
IterationRange = 6;         % Stellen-Genauigkeit

TextWFG = {'_Rect', '_LinR', '_LinF', '_Tri', '_Sin', '_SinInv', '_Gauß'};
%% --- Vorbereitung des McNeal-Modells
noMitte = 1 + floor(noNode/2);
Node = noMitte + floor(linspace(0, noMitte-1, nNode));
% --- Stabilitätsfaktor
rAxon = rhoAxon* LAxon;
Raxon = 4* rhoAxon* LAxon/(pi* dAxon^2);
tau = Raxon* Cmem* pi* dAxon* LAxon;
ath = LAxon^2/tau;
CFL = ath* dt0/LAxon.^2; 

switch(ModulationStim)
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

%% --- Parameter-Sweep (Parallel Computing)
noIte = zeros(size(Dis));
Qth = zeros(size(Dis));
Eth = zeros(size(Dis));
Ith = zeros(size(Dis));
PST = zeros(size(Dis));

parpool;
pctRunOnAll warning off
tic;
disp('.. Start!');
disp('-----------------------------------------------------');
cd 2_Funktionen
parfor k = 1:1:length(Dis)
    fstim = StimStart_f;
    Istim = StimStart_I;
    if(ModulationStim == 3)
        Iteration = 10^floor(log10(fstim))./logspace(0, 12, 13);
    else
        Iteration = 10^floor(log10(Istim))./logspace(0, 12, 13);
    end
    
    IterationNo = 1;
    IterationDeltaNo = 0;
    IterationDone = 0;
    IterationFirst = 1;
    IterationCycle = 1;
    cntDir = 0;
    
    %% --- Determining the Threshold
    while IterationDone == 0
        xyNode = 1e-6* [Dis(k) 0.5];
        % --- Medium-Widerstand bestimmen
        Rmed = zeros(1, noNode);
        for i = 1:1:noNode
            Rmed(i) = rhoMedium/(2*pi*norm(xyNode + [0 (i-noMitte)*LAxon] - xyElec));
        end
        Rtis = max(Rmed);
        % --- Erstellen der Anregung
        noIte(k) = noIte(k) +1;
        Tph = TphS;
        Tstim = Tph + phaseWFG*(Tph + Tipd); 
        Tmax = max([TSim (2*dT0 + Tstim)]);
        t = 0:dt0:Tmax;
        I_EL = zeros(size(t)); 

        % --- Erstellen der WFG
        dT = 0;
        T0 = t;
        I_EL0 = I_EL;
        for i = 1:1:phaseWFG+1           
            I0 = Istim* (-1)^(firstStim+i);
            A = find(t >= dT0 + dT);
            B = find(t >= dT0 + dT + Tph);
            Istim0 = waveformGenerator(patternWFG, I0, int16((B(1)-A(1)+1)*dt0/dt1));
            Istim1 = stimModulation(ModulationStim, dt1, Istim0, [Rtis Cdl], [Cmod fstim Qmod 200e-9]);
            I_EL0 = [I_EL(1:A(1)-1), Istim1, I_EL(B(1)+1:end)];
            I_EL = I_EL0;
            T1 = t(A(1)) + (t(B(1))-t(A(1)))/(length(Istim1)-1)*(0:1:length(Istim1)-1);
            T0 = [t(1:A(1)-1), T1, t(B(1)+1:end)];
            t = T0;
            dT = Tph + Tipd;
        end
        I_EL = I_EL + NoisePeak*(rand(size(t))-0.5);
        
        % --- Erstellen der Anregung (Teil 2)
        dt = [diff(T0), dt0];
        U_WE = zeros(size(t));
        E_WE = zeros(size(t));
        Q_dl = zeros(size(t));
        U_dl = zeros(size(t));

        % --- Arrays deklarieren
        Qinj = zeros(noNode, length(t));
        Einj = zeros(noNode, length(t));
        Vint = zeros(noNode, length(t));
        Vext = zeros(noNode, length(t));
        m = zeros(noNode, length(t)); 
        n = zeros(noNode, length(t)); 
        h = zeros(noNode, length(t));

        %% --- Computing Neural Network (McNeal-Modell)
        for i = 1:1:length(t)-1
            % --- Verhalten des Elektroden-Interfaces
            U_dl(i+1) = U_dl(i)*(1 - dt(i)/(Rfar* Cdl)) + I_EL(i)/Cdl*dt(i);
            Q_dl(i+1) = Cdl* U_dl(i);
            U_WE(i+1) = Rtis* I_EL(i) + U_dl(i);

            % --- Berechnung des extrazellulären Potentials am Node
            Vext(:, i) = Rmed.* I_EL(i); 

            % --- Berechnung der Ladungsmenge und Energie
            if(i > 1)
                E_WE(i+1) = E_WE(i-1) + max(Rmed)*(I_EL(i-1)).^2* dt(i);
                Qinj(:, i) = Qinj(:, i-1) + Vext(:, i-1)/Raxon* dt(i);
                Einj(:, i) = Einj(:, i-1) + (Vext(:, i-1)).^2/Raxon* dt(i);
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
        end  

        %% --- Nachbearbeitung
        Vint(:,1) = Vint(:,2);  
        Qinj(:,end) = Qinj(:,end-1);
        Einj(:,end) = Einj(:,end-1);

        Q0 = [min(Q_dl) max(Q_dl)];
        Q1 = [min(Qinj(noMitte, :)) max(Qinj(noMitte, :))];
        E0 = max(E_WE);

        %  --- Prüfen, ob Erregungsleitung statt fand (Ende der Nodes)
        [VmaxAP] = findpeaks(Vint(end,:), 'MinPeakHeight', 0);%, 'MinPeakDistance', 1.5e-3/dt0);  

        % --- Adaptive Anpassung der Anregung
        if(IterationFirst)
            cntDir = (-1)^(1+isempty(VmaxAP));
            IterationFirst = 0;
        end 
        plot(Vint(noMitte+1,:))
        if((isempty(VmaxAP) == 1-cntDir) || (isempty(VmaxAP) == 2+cntDir))
            if(ModulationStim == 3)
                fstim = fstim - (1+cntDir)/2* Iteration(IterationNo + IterationDeltaNo);
            else
                Istim = Istim - (1+cntDir)/2* Iteration(IterationNo + IterationDeltaNo);
            end
            IterationNo = IterationNo +1;
            IterationFirst = 1;
            if(IterationNo > IterationRange)||(IterationNo+IterationDeltaNo > length(Iteration))
                Ith(k) = Istim;
                fth(k) = fstim;
                Qth(k) = max(abs(Q0));
                Eth(k) = E0;
                [V0, x0] = max(Vint(noMitte, :));
                PST(k) = t(x0)-dT0;
                IterationDone = 1;
                disp(['.. Step ', num2str(k), ' of ', num2str(length(Dis)), ' done!']);
            else
                if(ModulationStim == 3)
                    fstim = fstim + 5* Iteration(IterationNo + IterationDeltaNo);
                else
                    Istim = Istim + 5* Iteration(IterationNo + IterationDeltaNo);
                end
            end
        else
            if(ModulationStim == 3)
                fstim = fstim + cntDir* Iteration(IterationNo + IterationDeltaNo);
                if(fstim <= 0)
                    fstim = fstim - cntDir*(Iteration(IterationNo + IterationDeltaNo) - Iteration(IterationNo + IterationDeltaNo +1));
                    IterationDeltaNo = IterationDeltaNo +1;
                end
            else
                Istim = Istim + cntDir* Iteration(IterationNo + IterationDeltaNo);
            end
        end  
        % --- Abbruch, wenn es zu lange läuft!
        IterationCycle = IterationCycle +1;
        if(IterationCycle >= 500)
            Ith(k) = 1;
            fth(k) = 1e-9;
            Qth(k) = 1;
            Eth(k) = 1;
            PST(k) = 1;
            IterationDone = 1;
            disp(['.. Step ', num2str(k), ' of ', num2str(length(Dis)), ' aborted!']);
        end
    end
end
delete(gcp);
clear A B;
clear m0 m1 n0 n1 h0 h1 Vint1 kT i;
toc

%% --- Daten und Bild speichern
% --- Pfad erstellen
PathFig = [pwd '\1_Bilder\'];
DateStr = datestr(now, 'yymmdd');
FileName = [DateStr, '_SimNeuron_'];
if(phaseWFG)
    FileName = [FileName 'Bi'];
else
    FileName = [FileName 'Mo'];
end
if(firstStim)
    FileName = [FileName 'Ano'];
else
    FileName = [FileName 'Cat'];
end
switch(ModulationStim)
    case 1
        FileName = [FileName 'VCS'];
    case 2
        FileName = [FileName 'CCS'];
    case 3
        FileName = [FileName 'ChCS'];
end
FileName = [FileName '_dSweep'];

% --- Daten speichern 
PathData = [pwd '\0_Daten\'];
save([PathData FileName '.mat'], 'Tipd', 'ModulationStim', 'firstStim', 'phaseWFG', 'patternWFG', 'T', 'Raxon', 'LAxon', 'noNode', 'Cmod', 'Cdl', 'Rfar', 'TphS', 'Ith', 'Qth', 'Eth');

%% --- Auswertung und Ausgabe
disp('-----------------------------------------------------');
disp('Auswertung');
disp(['Axon-Widerstand: ',      num2str(1e-3*Raxon),        ' kOhm']);

% --- Plotten
% --- Legenden
YString1 = ['E_{th}/Z / nJ'];
Legend = [];
switch(ModulationStim)
    case 1
        YString1 = [YString1 '; U_x / V'];
        Legend = 'U_{th}';
        corFac = 1;
        YLim = logspace(-3, 1, 5);

    case 2
        YString1 = [YString1 '; I_{th} / µA'];
        Legend = 'I_{th}';
        corFac = 1e6;
        YLim = logspace(-3, 2, 6);

    case 3
        YString1 = [YString1 '; f_{th} / MHz'];
        Legend = 'min(f_{mod})';
        corFac = 1e-6;
        YLim = logspace(-2, 1, 4);
        Ith = fth;
end

close all;
line = 1.5;
figure(1);
yyaxis left;
loglog(Dis, corFac*Ith, 'k', 'Linewidth', line);
hold on;    grid on;
loglog(Dis, 1e9*Eth, 'r', 'Linewidth', line);
ylabel(YString1);
ylim([YLim(1) 2*YLim(end)]);
yticks(YLim);

yyaxis right;
loglog(Dis, 1e9*abs(Qth), 'b', 'Linewidth', line);
grid on;

setGraphicStyle(1);
ylabel('Q_{th} / nC');
xlabel('d_{0} / µm');
legend({Legend, ' E_{th}', 'Q_{th}'}, 'Location', 'NorthEast');


% --- Bild speichern 
saveas(gcf, [PathFig FileName '.fig']); 
%saveas(gcf, [PathFig FileName '.eps']);
saveas(gcf, [PathFig FileName '.jpg']); 
disp('Fertig!');