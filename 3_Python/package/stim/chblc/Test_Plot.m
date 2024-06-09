close all;
clear all;
clc;

line = 1;

%% --- Berechnung
x = linspace(0,10,1000);
y1 = 5*x;
y2 = -1.6*(x-5).^2 + 40; 
y3 = 7.5*x;

mask = find(y2 >= y1);
fx = [x(mask), fliplr(x(mask))];
fy = [y1(mask), fliplr(y2(mask))];

%% --- Plotten
close all;

h = figure(1);

fh = fill(fx,fy, [196 196 204]/255);
uistack(fh, 'bottom');
hold on;    grid on;

plot(x, y1, 'k', 'Linewidth', line);
plot(x, y2, 'k', 'Linewidth', line);
plot(x, y3, 'r', 'Linewidth', line);
