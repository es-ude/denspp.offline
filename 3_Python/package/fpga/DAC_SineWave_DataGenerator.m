close all;
clear all;
clc;
%% --- Variablen
nBit = 7;
nCLK = 0:1:123;
fOSC = 10e6;

%% --- Berechnung
LSB = 1./(2^nBit-1);

N = 0:1.25:2^nBit-1;
Sine = round((2^nBit-1)*0.5* sin(2*pi*N/(2^nBit-1)))';
dQinj = trapz(Sine)/length(Sine);

%% --- Bestimmung
A = find(Sine == 2^(nBit-1)-1);
LengthQuarterSine = A(1) + ceil(size(N(A),2)/2)-1
fMod = fOSC/(4* LengthQuarterSine -4)
SineLUT = Sine(1:LengthQuarterSine);

CLK_out = 0.5*fOSC./(1 + nCLK)/(4* LengthQuarterSine -4);

%% --- Textausgabe für Verilog
disp('Registerbeschreibung in Verilog');
disp(strjoin(["localparam LENGTH_LUT = ", int2str(nBit-2), "'d", int2str(LengthQuarterSine-1), ";"], ''));
disp(strjoin({'wire [', int2str(nBit-2), ':0] SINE_LUT [LENGTH_LUT:0];'}, ''));
for i = 1:LengthQuarterSine
    disp(strjoin(["assign SINE_LUT[", int2str(i-1), "] = ", int2str(nBit-1), "'d", int2str(SineLUT(i)), ";"], ''));
end
disp('');
for i = 1:LengthQuarterSine
    disp(strjoin({'input [', int2str(nBit-2), ':0] SINE_LUT_', int2str(i-1)}, ''));
end
for i = 1:LengthQuarterSine
    disp(strjoin({'.SINE_LUT_', int2str(i-1),'(SINE_LUT[', int2str(i-1), ']),'}, ''));
end

for i = 1:LengthQuarterSine
    disp(strjoin(["assign SINE_LUT[", int2str(i-1), "] = SINE_LUT_", int2str(i-1), ";"], ''));
end

%% --- Plotten
figure(1);
stairs(Sine);
xlabel('xDAC'); ylabel('yDAC_SINE');

figure(2);
semilogy(nCLK, 1e-3*CLK_out, 'LineStyle', 'none', 'Marker', '.', 'MarkerSize', 8);
xlabel('xCLK'); ylabel('f_{OSC} / kHz');