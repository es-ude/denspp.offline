%% --- Header
    %Author:            A. Erbslöh
    %Erstelldatum:      30.05.2015
    %Aktualisierung:    04.07.2017
    %Company:           EBS
    %Version:           1v0
    %Projekt:           Modellierung einer selbstregelnden Motors
    %Bemerkungen:       activeFB noch fehlerbehaftet
    
%% --- Funktion der elektrischen Stim.- Methode

function [U_T, I_T, Q_T, P_v, E_v, Q_avg] = StimMethod(t, stim, n_stim, Fstim, Z_target, kChCS)
activeFB = 1;
k_an = kChCS(1);
k_ca = kChCS(2);
Tstim = kChCS(3);
fChCS = kChCS(4);
Cstim = kChCS(5);

Rtis = Z_target(1);
Cdl = Z_target(2);
Rfar = Z_target(3);

U_T = 0*t;
I_T = 0*t;
Q_T = 0*t;
Q_avg = 0;
dt = t(2)-t(1);
switch(stim)
    case 1
        Uref_VCS = -0.5*max(Fstim);
        %VCS-Methode        
        tau = Rtis* Cdl;
        T_min = 5* tau;
        k0 = 1 + dt/ tau;
        U_T(1) = Fstim(1)*dt/k0; 
        for i=2:1:length(t)
            U_T(i) = (Fstim(i)*dt/tau + U_T(i-1))/k0;
        end
        I_T = Cdl* diff(U_T)./diff(t);
        I_T(end+1) = I_T(end);
        for i = 2:1:length(t)
            Q_T(i) = Q_T(i-1) + I_T(i-1)* dt;
        end      
        P_v = I_T.*(Fstim+Uref_VCS);      
    case 2
        %CCS-Methode
        I_T = Fstim;
        for i = 2:1:length(t)
            Q_T(i) = Q_T(i-1) + I_T(i-1)* dt;
        end
        U_T = Rtis* I_T + Q_T/Cdl;
        P_v = U_T.* I_T;    
      
    case 3
        n_Puls = floor(fChCS* (k_an + k_ca)* Tstim);
        Reff = Rfar* Rtis/(Rfar + Rtis);
        Ceff = (Cdl* Cstim)/(Cdl + Cstim);
        disp(['max. Mod.-Frequenz: ' num2str(1e-6/(10* Reff* Ceff)) ' MHz']);
        disp(['Anzahl von Pulsen: ', num2str(n_Puls)]);
        
        %ChCS-Methode
        tau = Reff* Ceff;
        k0 = 1 + dt/ tau;
        %Erstellen der Abtastpunkte in Abhängigkeit der Spannung
        state_an = 0*t;
        state_ca = 0*t;
        z = find(Fstim > 0);     state_an(z) = 1;    z = 0;
        z = find(Fstim < 0);     state_ca(z) = 1;    z = 0;
                
        sig_ChCS = (state_ca + state_an)/2.* (1 + square(2*pi*fChCS*t));
        dsig = diff(sig_ChCS);
        dsig(end+1) = 0;
        
        U_T0 = zeros(size(t));
        
        x_sample = 0*t;   
        p2sample = find(dsig == -1);
        p1sample = find(dsig == 1);  
        x_sample(p1sample(1):p2sample(1)) = Fstim(p2sample(1));
        pdsample = mean(p2sample - p1sample);
        for i = 2:1:length(p2sample)
            x_sample(p1sample(i):p2sample(i)) = Fstim(p2sample(i)) - activeFB* U_T(p1sample(i-1));
            
            %Berechnung des Stromverlaufs nach DGL
            for k = 1:1:pdsample
                U_T0(p1sample(i)+k+1) = U_T0(p1sample(i)+k)*(1 - dt/tau) + x_sample(p1sample(i)+k)*dt/tau;
            end        
            I_T(p1sample(i):p2sample(i)-1) = Ceff* diff(U_T0(p1sample(i):p2sample(i)))/dt;
            
            z = find(I_T.* state_an < 0);   I_T(z) = 0;
            z = find(I_T.* state_ca > 0);   I_T(z) = 0;
            
            %Berechnung der injizierten Ladungsmenge
            k = 0;
            while(t(end) ~= t(p1sample(i)+k))%for k = 0:1:length(t)-1
                Q_T(p1sample(i)+k+1) = Q_T(p1sample(i)+k) + I_T(p1sample(i)+k)* dt;
                k = k +1;
            end
            Q_T(p2sample(i):end) = Q_T(p2sample(i));
            
            U_T(p1sample(i):p2sample(i)) = Q_T(p1sample(i):p2sample(i))/(Cdl + Cstim) + Rtis* I_T(p1sample(i):p2sample(i));
            U_T(p2sample(i)+1:end) = U_T(p2sample(i));
        end 
        
        % --- injizierte Ladungsmenge pro Stim.-Puls
        p1sample = find((dsig) == 1);
        Q_ph = diff(Q_T(p1sample));
        Q_ph(find(Q_ph == 0)) = [];
        l1 = length(Q_ph);
        Q_avg = Q_ph(1:l1/n_stim);

        %Leistungsaufnahme und Energie berechnen
        P_v = U_T.* I_T;    
end
%E_v = cumtrapz(t, ((P_v)));
E_v = trapz(t, abs(real(P_v)));
disp(['Leistung: ' num2str(1e6* E_v(end)/(n_stim* Tstim)) ' µW']);
disp(['Energie: ' num2str(1e9* E_v(end)/ n_stim) ' nWs']);
