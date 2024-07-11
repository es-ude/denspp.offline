% Dependencies:
% 1. activation/2_Funktionen/waveformGenerator.m
% 2a. activation/2_Funktionen/stimModulation.m
% 2b. sim/stimMethod.m

close all;
clearvars;

f_samp = 1e9;
t_samp = 1 / f_samp;
f_sine = 1e3;
t_sine = 1 / f_sine;
t_rest = t_sine;

t_sineh = t_sine / 2;
f_sineh = 1 / t_sineh;
hw_len = t_sineh / t_samp;
r_len = t_rest / t_samp;
rep = 20;

[wf_sineh, signame] = waveformGenerator(5,1,hw_len);
wf_sine = [-wf_sineh wf_sineh];
wf_pulse = [wf_sine zeros(1, r_len)];
wf = repmat(wf_pulse, 1, rep);

num_samples = length(wf);
t = t_samp* (0:1:num_samples-1);

% figure(1);
% plot(t, wf);
% ylabel('amp');
% % setGraphicStyle(0);
% ylim([-1.5 1.5]);
% grid;
% yticks(-1.5:0.5:1.5);
% xticks(0:5e-4:t_samp*num_samples);

Rtis = 20e3;
Cdl = 6e-9;
Rfar = 10e-6;
amp = 10e-6;

% [I_T, U_T, Q_T, E_v, P_v] = stimModulation(2, t_samp, amp .* wf, [Rtis Cdl], zeros(1, 4));
[U_T, I_T, Q_T, P_v, E_v, Q_avg] = StimMethod(t, 2, rep, amp .* wf, [Rtis Cdl Rfar], zeros(1, 5));

figure(2);
subplot(3,1,1);
plot(t,U_T);
subplot(3,1,2);
plot(t,I_T);
subplot(3,1,3);
plot(t,Q_T);

% Results: CCS calculation does not consider leakage/discharge current at Rfar


