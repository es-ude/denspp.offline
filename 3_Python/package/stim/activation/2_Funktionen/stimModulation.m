%% --- Header
    %Author:            A. Erbslöh
    %Erstelldatum:      30.05.2015
    %Aktualisierung:    08.06.2020
    %Company:           EBS
    %Version:           1v2
    %Projekt:           Modellierung der funktionalen Stimulations-Modulation
    
%% --- Funktion der elektrischen Stim.- Methode
function [I_T, U_T, Q_T, E_v, P_v] = stimModulation(stim, dt, Fstim, Z_target, ChCS_Data)
Rtis = Z_target(1);
Cdl = Z_target(2);
Rfar = 2e6;
Stimuli = [Fstim Fstim(end)];

t = dt* (0:1:length(Fstim)-1);
U_T = 0*t;
I_T = 0*t;
Q_T = 0*t;

%% --- Modulationen
switch(stim)
    case 1
        %% --- VCS-Methode    
        %disp('Spannungsgesteuerte Stimulation');
        for i=2:1:length(t)
            U_T(i) = U_T(i-1)*(1 - dt/(Rtis*Cdl)) + Stimuli(i)*dt/(Rtis* Cdl);
        end
        I_T = Cdl* diff(U_T)./diff(t);
        I_T(end) = I_T(end);
    case 2
        %% --- CCS-Methode
        %disp('Stromgesteuerte Stimulation');
        I_T(1:end) = Fstim; 
        I_T(end) = 0;
    case 3
        %% --- ChCS-Methode
        Cmod = ChCS_Data(1);
        fmod = ChCS_Data(2);
        Qmod = ChCS_Data(3);
        Tau0 = ChCS_Data(4);
        
        %disp('Ladungsgesteuerte Stimulation');
        n_Puls = 1+floor(fmod* t(end));
        Reff = Rfar* Rtis/(Rfar + Rtis);
        Ceff = (Cdl* Cmod)/(Cdl + Cmod);
        tau = Reff* Ceff;
        %disp(['Anzahl von Takten: ' num2str(n_Puls)]);
        %disp(['max. Mod.-Frequenz: ' num2str(1e-6/(10* Reff* Ceff)) ' MHz']);
        
        % Erstellen der exp-Waveform
        dx = floor(length(t)/(n_Puls));
        IQ = exp(-(0:dx-1)*dt./Tau0);

        %Erstellen der Abtastpunkte in Abhängigkeit der Spannung
        dX = 1+(0:1:n_Puls-1)*dx;
        I0 = Qmod/(tau*(1-exp(-5)))* Fstim(dX)/max(abs(Fstim));    
        
        %Berechnung des Stromverlaufs nach DGL
        for k = 1:1:n_Puls
            I_T(dX(k):(dX(k)+length(IQ))-1) = I0(k)*IQ;
        end
end
%% --- Ladungsintegration und Energieberechnung
for i = 2:1:length(t)
    Q_T(i) = Q_T(i-1) + I_T(i-1)* dt;
end   
% Berechnung der Spannung
if(stim > 1)
    U_T = Rtis* I_T + Q_T/Cdl;     
end

P_v = U_T.* I_T* 1e-9; 
E_v = trapz(t, abs(real(P_v)));
%disp(['Ladungsmenge: '  num2str(Q_T(end))               ' nC']);
%disp(['Leistung: '      num2str(1e6* E_v(end)/t(end))   ' µW']);
%disp(['Energie: '       num2str(1e6* E_v(end))          ' µWs']);
