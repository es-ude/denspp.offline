close all;  clear all;  clc;
warning('off');
line = 1.5;

%% --- Parameter
n = 0.74;
U_AP = 80e-6;   f_AP = 50;
U_LFP = 1e-3;   f_LFP = 7.25;
U_off = 100e-6;

Tend = 1;
dt = 10e-6;
fHP = 100;

%% --- Signale generieren
t = (0:dt:Tend);
% --- Einsetzen des LFP
LFP = zeros(size(t));
for i=1:1:4
   LFP = LFP + exp(-(i-1)^2*0.6)*sin(2*pi*f_LFP*i.*t); 
end
LFP = U_LFP* LFP./max(LFP);

% --- Einsetzen der AP
AP = zeros(size(t));
nAP = floor(t(end)*f_AP);
dT_AP = floor(2e-3/dt)-1;

%AP0 = -U_AP*exp(-(0:1:dT_AP)*dt/0.2e-3) + U_AP/4*sin(pi*(0:1:dT_AP)/dT_AP);
cor = 1;        AP0 = n* exp(-(((0:1:dT_AP)*dt-0.2e-3)/0.2e-3).^2) + (n-1)* sin(pi*(0:1:dT_AP)/dT_AP);
cor = max(AP0); AP0 = -(n/cor* exp(-(((0:1:dT_AP)*dt-0.2e-3)/0.2e-3).^2) + (n-1)* sin(pi*(0:1:dT_AP)/dT_AP));

for i=3:1:nAP
    x0 = (i-1)*length(t)/nAP + 1;
    x1 = x0 + dT_AP;
    AP(x0:x1) = U_AP* AP0; 
end

[b, a] = butter(2, 2*dt*[10 5e3], 'bandpass');
AP = filter(b,a, AP);

% --- Thermisches Rauschen
Un = wgn(1, length(AP), -100);
Unoise = rms(Un)
SNR = (U_AP* rms(AP0)/Unoise).^2

% --- Signalzusammensetzen
U_in = AP + LFP + U_off + Un;

%% --- NEO anwenden
[b, a] = butter(1, 2*dt*fHP, 'high');
Udiff1 = 10* filter(b,a, U_in);
Udiff2 = 10* filter(b,a, Udiff1);

U_NEO = (Udiff1).^2 - Udiff2.* U_in;
U_ED = (Udiff1).^2;
U_eED = (Udiff2).^2;

[b, a] = butter(1, 2*dt*5e3, 'low');
U_NEO = filter(b,a, U_NEO);
U_ED = filter(b,a, U_ED);
U_eED = filter(b,a, U_eED);

%% --- Schwellwert-Detektion
X0 = 20e-3/dt;
normAP = abs(AP)/U_AP;
normNEO = U_NEO/max(U_NEO(X0:end));
normED = U_ED/max(U_ED(X0:end));
normeED = U_eED/max(U_eED(X0:end));

ThNEO = 0.5./(1 + exp(-0.25*Unoise./U_AP));
ThED = 0.5./(1 + exp(-0.25*Unoise./U_AP));
TheED = 0.5./(1 + exp(-0.25*Unoise./U_AP));
    
Thres1 = floor(0.5*length(t)/nAP);
Thres2 = floor(dT_AP/100);

[Xth_AP, xAP]   = (findpeaks(normAP(X0:end),    'MinPeakHeight', 0.6,   'MinPeakDistance', Thres1, 'MinPeakWidth', Thres2));
[Xth_NEO, xNEO] = (findpeaks(normNEO(X0:end),   'MinPeakHeight', ThNEO, 'MinPeakDistance', Thres1, 'MinPeakWidth', Thres2));
[Xth_ED, xED]   = (findpeaks(normED(X0:end),    'MinPeakHeight', ThED,  'MinPeakDistance', Thres1, 'MinPeakWidth', Thres2));
[Xth_eED, xeED] = (findpeaks(normeED(X0:end),   'MinPeakHeight', TheED, 'MinPeakDistance', Thres1, 'MinPeakWidth', Thres2));

%% --- Detektion der True-Positiv und False-Negative
DT_NEO = 0; TR_NEO = 0; FR_NEO = 0;
DT_ED = 0;  TR_ED = 0;  FR_ED = 0;
DT_eED = 0; TR_eED = 0; FR_eED = 0;

% --- NEO-Auswertung
for k = 1:1:length(xNEO)
    detected = 1;
    for i=1:1:length(xAP)
       if((xNEO(k) >= (xAP(i)-dT_AP/2))&&(xNEO(k) <= (xAP(i)+dT_AP/2)) && detected)
           TR_NEO = TR_NEO +1;
           detected = 0;
       end
    end
    if(detected)
       FR_NEO = FR_NEO + 1;
    end
end
DT_NEO = TR_NEO/length(xAP);
TR_NEO = TR_NEO/length(xNEO);
FR_NEO = FR_NEO/length(xNEO);

% --- ED-Auswertung
for k = 1:1:length(xED)
    detected = 1;
    for i = 1:1:length(xAP)
       if((xED(k) >= (xAP(i)-dT_AP/2))&&(xED(k) <= (xAP(i)+dT_AP/2))&& detected)
           TR_ED = TR_ED +1;
           detected = 0;
       end
    end
    if(detected)
       FR_ED = FR_ED + 1;
    end
end
DT_ED = TR_ED/length(xAP);
TR_ED = TR_ED/length(xED);
FR_ED = FR_ED/length(xED);

% --- eED-Auswertung
for k = 1:1:length(xeED)
    detected = 1;
    for i = 1:1:length(xAP)
       if((xeED(k) >= (xAP(i)-dT_AP/2))&&(xeED(k) <= (xAP(i)+dT_AP/2))&& detected)
           TR_eED = TR_eED +1;
           detected = 0;
       end
    end
    if(detected)
       FR_eED = FR_eED + 1;
    end
end
DT_eED = TR_eED/length(xAP);
TR_eED = TR_eED/length(xeED);
FR_eED = FR_eED/length(xeED);

%% --- Plotten
close all;
figure(1);
yyaxis left;
plot(t, 1e6* (AP+Un), 'k', 'linewidth', line);
setGraphicStyle(0);
ylabel('U_{AP} / µV');

yyaxis right;
plot(t, 1e3* (U_in), 'r', 'linewidth', line);
setGraphicStyle(1);
ylabel('U_{EL} / mV');
xlabel('Zeit t / s');
legend({'U_{AP}', 'U_{EL}'}, 'Location', 'South');

figure(2);
subplot(3,1,1);
plot(t, normNEO, 'k', 'linewidth', line/2);   
setGraphicStyle(0);
ylim([-0.05 1.05]);
yticks(0:0.2:1);
ylabel('norm(U_{NEO})');

subplot(3,1,2);
plot(t, normED, 'k', 'linewidth', line/2);
setGraphicStyle(0);
ylim([-0.05 1.05]);
yticks(0:0.2:1);
ylabel('norm(U_{ED})');

subplot(3,1,3);
plot(t, normeED, 'k', 'linewidth', line/2);
setGraphicStyle(1);
ylim([-0.05 1.05]);
yticks(0:0.2:1);
xlabel('Zeit t / s');
ylabel('norm(U_{eED})');
