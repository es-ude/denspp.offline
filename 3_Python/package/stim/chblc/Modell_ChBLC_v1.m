close all;
clear all;
clc;

line = 1;

%% --- Modellparameter
noStim = 5000;
xStimPlot = 50;

Tano_Norm = 80;
Tano_SD = 1;
dQcat_Norm = 18;
dQano_Norm = 18;
dQ_Std = 1;

%% --- Berechnung
x = 0:1:noStim-1;

Qre = 0;
Qcat = 0;
Tano_re = 0;

dQcatM = 0;
dQanoM = 0;
for i = 1:1:noStim
   dQcat = randn(Tano_Norm,1)*dQ_Std + dQcat_Norm;
   dQano = randn(Tano_Norm+20,1)*dQ_Std + dQano_Norm;
   
   dQcatM(i) = mean(dQcat);
   dQanoM(i) = mean(dQano);
   
   Qcat(i) = Qre(i);
   for k = 1:1:Tano_Norm
       Qcat(i) =  + Qcat(i) - dQcat(k);
   end
   k = 0;
   while(Qcat(i) <= 0)
       k = k + 1;
       Qcat(i) = Qcat(i) + dQano(k);
   end
   Qre(i+1) = Qcat(i);
   Tano_re(i) = k;
end
Qre(end) = [];
Tano_id = 0.*x + Tano_Norm;

% --- Restladungen
dQ = mean(abs(dQcatM-dQanoM))* Tano_Norm;
y1 = -dQ*x;
y2 = +dQ*x; 
y3 = 0.005*x;
y4 = 0.*x + 1500;
y5 = 0.*x - 1500;

mask = find(y2 >= y1);
fx = [x(mask), fliplr(x(mask))];
fy = [y1(mask), fliplr(y2(mask))];

%% --- Plotten (
PathFig = [pwd '\'];

close all;
h = figure(1);
set(h, 'Units', 'centimeters', 'Position', [1 2 15 10]);

yyaxis left;
fh = fill(fx,fy, [196 196 204]/255, ...
    'FaceAlpha',0.4,...
    'EdgeAlpha',0,...
    'LineStyle','--');
uistack(fh, 'bottom');
hold on;    grid on;

%plot(x, y1, 'k--', 'Linewidth', line);
%plot(x, y2, 'k--', 'Linewidth', line);
plot(x, Qre, 'k', ...
    'LineStyle', 'none', ...
    'Marker', '.', ...
    'MarkerSize', 8);
%plot(x, y4, 'r--', 'Linewidth', line);
%plot(x, y5, 'r--', 'Linewidth', line);

ylabel('Residual charges \Delta Q [pC]');
ylim([-40 40]);
yticks(-40:10:40);
xlim([0 xStimPlot]);

yyaxis right;
plot(x, Tano_id, 'b--', 'Linewidth', line);
plot(x, Tano_re, 'r', ...
    'LineStyle', 'none', ...
    'Marker', '.', ...
    'MarkerSize', 8);
xlim([0 xStimPlot]);
ylim([75 82]);

setGraphicStyle();
ylabel('No. of clocks n_{ano}');
xlabel('No. of Stim.-Pulses');
legend({'\Delta Q (wo Regulation)';'\Delta Q (w Regulation)';'n_{ano} (ideal)';'n_{ano} (real)'}, 'Location', 'SouthEast');

% --- Speichern
savefig(h, [PathFig 'TranModell.fig']);
saveas(h, [PathFig 'TranModell.eps'], 'eps');
saveas(h, [PathFig 'TranModell.jpg'], 'jpg');

% --- Plotten (Histogramm)
h = figure(2);
set(h, 'Units', 'centimeters', 'Position', [17 2 15 10]);

subplot(1,2,1);
histfit(dQcatM, 20);
grid on;
xlabel('Q_{ph,cat} [pC]');
setGraphicStyle();

x_axis = xlim;
dx = diff(xlim)/16;
y_axis = ylim;
dy = diff(ylim)/16;
text(x_axis(1)+dx, y_axis(2)-dy,        ['Mean =' num2str(round(mean(dQcatM),3))]);
text(x_axis(1)+dx, y_axis(2)-dy*(1.6),  ['Std =' num2str(round(std(dQcatM),3))]);

subplot(1,2,2);
histfit(dQanoM, 20);
grid on;
xlabel('Q_{ph,ano} [pC]');
setGraphicStyle();

x_axis = xlim;
dx = diff(xlim)/16;
y_axis = ylim;
dy = diff(ylim)/16;
text(x_axis(1)+dx, y_axis(2)-dy,        ['Mean =' num2str(round(mean(dQanoM),3))]);
text(x_axis(1)+dx, y_axis(2)-dy*(1.6),  ['Std =' num2str(round(std(dQanoM),3))]);

% --- Speichern
savefig(h, [PathFig 'Histo1.fig']);
saveas(h, [PathFig 'Histo1.eps'], 'eps');
saveas(h, [PathFig 'Histo1.jpg'], 'jpg');

% --- Plotten (Histogramm 2)
h = figure(3);
set(h, 'Units', 'centimeters', 'Position', [33 2 15 10]);

subplot(1,2,1);
histfit(Qre, 20);
grid on;
xlabel('\Delta Q [pC]');
setGraphicStyle();

x_axis = xlim;
dx = diff(xlim)/16;
y_axis = ylim;
dy = diff(ylim)/16;
text(x_axis(1)+dx, y_axis(2)-dy,        ['Mean =' num2str(round(mean(Qre),3))]);
text(x_axis(1)+dx, y_axis(2)-dy*(1.6),  ['Std =' num2str(round(std(Qre),3))]);

subplot(1,2,2);
histfit(Tano_re, 20);
grid on;
xlabel('n_{ano}');
setGraphicStyle();

savefig(h, [PathFig 'Histo2.fig']);
saveas(h, [PathFig 'Histo2.eps'], 'eps');
saveas(h, [PathFig 'Histo2.jpg'], 'jpg');

%% --- External Functions
function setGraphicStyle(changePos)
sizeText = 18;
dataFig = gcf;
dataSubPlot = findobj(allchild(dataFig), 'flat', 'Type', 'axes');
axesPos = get(dataSubPlot, 'Position');
noAxes = size(axesPos);

dataAxis = gca;
if(length(dataAxis.YAxis) == 2)
    dataAxis.YAxis(1).Color = 'k'; 
    dataAxis.YAxis(2).Color = 'k';
end
 
if(changePos)
    % --- Manipulation der Figure-Größe
    % normale Plots in 4:3-Format
    sizeFig = [1 2 16 12];
    switch(noAxes(1))
        case 3
            sizeFig = [1 2 16 15];
        case 4
            sizeFig = [1 2 16 20];
    end
    set(dataFig, 'Units', 'centimeters', 'Position', sizeFig);
    
    % --- Manipulation der Subplot-Größen
    if(length(dataAxis.YAxis) == 2)
        % --- Änderung bei Bild mit zwei Y-Achsen
        switch(noAxes(1))
            case 1
                dataSubPlot(1).Position = [0.13 0.17 0.72 0.82];
            case 2
                dataSubPlot(2).Position = [0.13 0.61 0.76 0.36];
                dataSubPlot(1).Position = [0.13 0.14 0.76 0.36];
            case 3
                dataSubPlot(3).Position = [0.13 0.74 0.76 0.24];
                dataSubPlot(2).Position = [0.13 0.42 0.76 0.24];
                dataSubPlot(1).Position = [0.13 0.1 0.76 0.24];
            case 4
                dataSubPlot(4).Position = [0.13 0.795 0.76 0.18];
                dataSubPlot(3).Position = [0.13 0.56  0.76 0.18];
                dataSubPlot(2).Position = [0.13 0.325 0.76 0.18];
                dataSubPlot(1).Position = [0.13 0.1  0.76 0.18];
        end
    else
        % --- Änderung bei Bild mit einer Y-Achse
        switch(noAxes(1))
            case 1
                dataSubPlot(1).Position = [0.13 0.14 0.82 0.82];
            case 2
                dataSubPlot(2).Position = [0.13 0.61 0.82 0.36];
                dataSubPlot(1).Position = [0.13 0.14 0.82 0.36];
            case 3
                dataSubPlot(3).Position = [0.13 0.74 0.82 0.24];
                dataSubPlot(2).Position = [0.13 0.42 0.82 0.24];
                dataSubPlot(1).Position = [0.13 0.1  0.82 0.24];                
            case 4
                dataSubPlot(4).Position = [0.13 0.795 0.81 0.18];
                dataSubPlot(3).Position = [0.13 0.56  0.81 0.18];
                dataSubPlot(2).Position = [0.13 0.325 0.81 0.18];
                dataSubPlot(1).Position = [0.13 0.1  0.81 0.18];
        end
    end
end

% --- Änderung der Achseneigenschaften
set(gca,'TickDir', 'out', ...                                                      %Die Ticks nach außen setzen
    'XMinorTick', 'on', 'YMinorTick', 'on', 'ZMinorTick', 'on', ...                %Minor Ticks einschalten
    'XMinorGrid', 'on', 'YMinorGrid', 'off', 'ZMinorGrid', 'on', ...                %Minor Ticks einschalten
    'Fontsize', sizeText, ...
    'FontName','Times New Roman', ...
    'XTickLabel', strrep(cellstr(get(gca, 'XTickLabel')), '.', ','), ...           %Dezimalpunkte durch Kommas ersetzen
    'YTickLabel', strrep(cellstr(get(gca, 'YTickLabel')), '.', ','), ...
    'XTickMode', 'auto', 'XTickLabelMode', 'auto', ...                             %belässt den Abstand der Ticks gleich, aber aktualisiert die Beschriftung automatisch
    'YTickMode', 'auto', 'YTickLabelMode', 'auto', ...
    'TickDir', 'out');  
end