close all;  clear all;  clc;
warning('off');
line = 1;

%% --- Parameter
U_AP = (1e-6:1e-6:100e-6);
n = 0.74;           f_AP = 50;
U_LFP = 2.6e-3;     f_LFP = 7.25;
U_off = 100e-6;

Tend = 25;
dt = 10e-6;
fHP = 100;

%% --- Sweep vorbereiten
DT_NEO = zeros(size(U_AP)); TR_NEO = 0*DT_NEO;    FR_NEO = 0*DT_NEO;
DT_ED = zeros(size(U_AP));  TR_ED = 0*DT_ED;      FR_ED = 0*DT_ED;
DT_eED = zeros(size(U_AP)); TR_eED = 0*DT_eED;    FR_eED = 0*DT_eED;
SNR = zeros(size(U_AP));

% --- Signale vordefinieren
t = (0:dt:Tend);
Un = wgn(1, length(t), -100);
Unoise = rms(Un);

nAP = floor(t(end)*f_AP);
dT_AP = floor(2e-3/dt)-1;
%               AP0 = -exp(-(0:1:dT_AP)*dt/0.2e-3) + U_AP/4*sin(pi*(0:1:dT_AP)/dT_AP);
cor = 1;        AP0 = n* exp(-(((0:1:dT_AP)*dt-0.2e-3)/0.2e-3).^2) + (n-1)* sin(pi*(0:1:dT_AP)/dT_AP);
cor = max(AP0); AP0 = -(n/cor* exp(-(((0:1:dT_AP)*dt-0.2e-3)/0.2e-3).^2) + (n-1)* sin(pi*(0:1:dT_AP)/dT_AP));

%% --- Start
enableParallelComputing(1,1);
pctRunOnAll warning off
parfor k = 1:1:length(U_AP)
    %% --- Signale generieren
    AP = zeros(size(t));
    LFP = zeros(size(t));
    for i=1:1:4
       LFP = LFP + exp(-(i-1)^2*0.6)*sin(2*pi*f_LFP*i.*t); 
    end
	LFP = U_LFP* LFP/max(LFP);
    
    for i=3:1:nAP
        x0 = (i-1)*length(t)/nAP + 1;
        x1 = x0 + dT_AP;
        AP(x0:x1) = U_AP(k)* AP0; 
    end

    [b, a] = butter(2, 2*dt*[10 5e3], 'bandpass');
    AP = filter(b,a, AP);

    % --- Signalzusammensetzen
    U_in = AP + LFP + U_off + Un;

    %% --- NEO anwenden
    [b, a] = butter(1, 2*dt*fHP, 'high');
    Udiff1 = 10* filter(b,a, U_in);
    Udiff2 = 10* filter(b,a, Udiff1);

    U_NEO = (Udiff1).^2 - Udiff2.* U_in;
    U_ED = (Udiff1).^2;
    U_eED = (Udiff2).^2;

    [b, a] = butter(2, 2*dt*5e3, 'low');
    U_NEO = filter(b,a, U_NEO);
    U_ED = filter(b,a, U_ED);
    U_eED = filter(b,a, U_eED);    

    %% --- Schwellwert-Detektion
    X0 = 20e-3/dt;
    normAP = abs(AP)/U_AP(k);
    normNEO = U_NEO/max(U_NEO(X0:end));
    normED = U_ED/max(U_ED(X0:end));
    normeED = U_eED/max(U_eED(X0:end));
    
    ThNEO = 0.5./(1 + exp(-0.25*Unoise./U_AP(k)));
    ThED = 0.5./(1 + exp(-0.25*Unoise./U_AP(k)));
    TheED = 0.5./(1 + exp(-0.25*Unoise./U_AP(k)));
    
    Thres1 = floor(0.5*length(t)/nAP);
    Thres2 = floor(dT_AP/100);

    [Xth_AP, xAP]   = (findpeaks(normAP(X0:end),    'MinPeakHeight', 0.6,   'MinPeakDistance', Thres1, 'MinPeakWidth', Thres2));
    [Xth_NEO, xNEO] = (findpeaks(normNEO(X0:end),   'MinPeakHeight', ThNEO, 'MinPeakDistance', Thres1, 'MinPeakWidth', Thres2));
    [Xth_ED, xED]   = (findpeaks(normED(X0:end),    'MinPeakHeight', ThED,  'MinPeakDistance', Thres1, 'MinPeakWidth', Thres2));
    [Xth_eED, xeED] = (findpeaks(normeED(X0:end),   'MinPeakHeight', TheED, 'MinPeakDistance', Thres1, 'MinPeakWidth', Thres2));
    
    %% --- Detektion der True-Positiv und False-Negative
    % --- NEO-Auswertung
    TR0 = 0;    FR0 = 0;
    for k0 = 1:1:length(xNEO)
        detected = 1;
        for i=1:1:length(xAP)
           if((xNEO(k0) >= (xAP(i)-dT_AP/2))&&(xNEO(k0) <= (xAP(i)+dT_AP/2)) && detected)
               TR0 = TR0 +1;
               detected = 0;
           end
        end
        if(detected)
           FR0 = FR0 + 1;
        end
    end
    DT_NEO(k) = TR0/length(xAP);
    TR_NEO(k) = TR0/length(xNEO);
    FR_NEO(k) = FR0/length(xNEO);

    % --- ED-Auswertung
    TR0 = 0;    FR0 = 0;
    for k0 = 1:1:length(xED)
        detected = 1;
        for i = 1:1:length(xAP)
           if((xED(k0) >= (xAP(i)-dT_AP/2))&&(xED(k0) <= (xAP(i)+dT_AP/2))&& detected)
               TR0 = TR0 +1;
               detected = 0;
           end
        end
        if(detected)
           FR0 = FR0 + 1;
        end
    end
    DT_ED(k) = TR0/length(xAP);
    TR_ED(k) = TR0/length(xED);
    FR_ED(k) = FR0/length(xED);

    % --- eED-Auswertung
    TR0 = 0;    FR0 = 0;
    for k0 = 1:1:length(xeED)
        detected = 1;
        for i = 1:1:length(xAP)
           if((xeED(k0) >= (xAP(i)-dT_AP/2))&&(xeED(k0) <= (xAP(i)+dT_AP/2))&& detected)
               TR0 = TR0 +1;
               detected = 0;
           end
        end
        if(detected)
           FR0 = FR0 + 1;
        end
    end
    DT_eED(k) = TR0/length(xAP);
    TR_eED(k) = TR0/length(xeED);
    FR_eED(k) = FR0/length(xeED);
    
    SNR(k) = (rms(U_AP(k)* AP0)/Unoise).^2;
    
    disp(['Schritt ', num2str(k), ' fertig!']);
end
enableParallelComputing(2,0);

%% --- Plotten
ID = 0*SNR + 1;

close all;
figure(1);
subplot(2,1,1);
plot(SNR, ID, 'k--', 'linewidth', line);    hold on;
plot(SNR, DT_NEO, 'k', 'linewidth', line);
plot(SNR, DT_ED, 'r', 'linewidth', line);
plot(SNR, DT_eED, 'b', 'linewidth', line);

setGraphicStyle(0);
ylim([-0.1 1.1]);
ylabel('Detektionsrate');
legend({'ideal', 'NEO', 'ED', 'eED'}, 'Location', 'SouthEast');

subplot(2,1,2);
plot(SNR, TR_NEO, 'k', 'linewidth', line);    hold on;
plot(SNR, FR_NEO, 'k--', 'linewidth', line);
plot(SNR, TR_ED, 'r', 'linewidth', line);
plot(SNR, FR_ED, 'r--', 'linewidth', line);
plot(SNR, TR_eED, 'b', 'linewidth', line);
plot(SNR, FR_eED, 'b--', 'linewidth', line);

setGraphicStyle(1);
ylim([-0.1 1.1]);
xlabel('SNR');
ylabel('Richtig/Falsch-Rate');
legend({'TR_{NEO}', 'FR_{NEO}', 'TR_{ED}', 'FR_{ED}', 'TR_{eED}', 'FR_{eED}'}, 'Location', 'East', 'Fontsize', 9);

%% --- Funktionen
% --- Funktion zur Einrichtung von Parallel Computing
function enableParallelComputing(state, enable)
    if(enable)
        if(state == 1)
            out = isempty(gcp('nocreate'));
            if(~out)
                disp(['... ParPool is already active']);
            else
                NumThreads = maxNumCompThreads;
                disp(['... ParPool is not activated (maximum of ', num2str(NumThreads),' workers)']);
                
                parpool(NumThreads); 
            end
        elseif(state == 2)
           delete(gcp)
        end
    end
end
