%% --- Header
    %Author:            A. Erbslöh
    %Erstelldatum:      30.05.2015
    %Aktualisierung:    28.05.2020
    %Company:           EBS
    %Version:           2v0
    %Projekt:           Änderung der Eigenschaften eines Figures
    %Bemerkungen:       /
    
%% --- Variablen
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
                    dataSubPlot(1).Position = [0.13 0.17 0.72 0.79];
                case 2
                    dataSubPlot(2).Position = [0.13 0.62 0.76 0.32];
                    dataSubPlot(1).Position = [0.13 0.15 0.76 0.32];
                case 3
                    dataSubPlot(3).Position = [0.13 0.14 0.76 0.32];
                    dataSubPlot(2).Position = [0.13 0.62 0.76 0.32];
                    dataSubPlot(1).Position = [0.13 0.15 0.76 0.32];
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
                    dataSubPlot(1).Position = [0.16 0.16 0.8 0.8];
                case 2
                    dataSubPlot(2).Position = [0.14 0.61 0.82 0.36];
                    dataSubPlot(1).Position = [0.14 0.14 0.82 0.36];
                case 3
                    dataSubPlot(3).Position = [0.14 0.74 0.82 0.24];
                    dataSubPlot(2).Position = [0.14 0.42 0.82 0.24];
                    dataSubPlot(1).Position = [0.14 0.1  0.82 0.24];                
                case 4
                    dataSubPlot(4).Position = [0.15 0.795 0.81 0.18];
                    dataSubPlot(3).Position = [0.15 0.56  0.81 0.18];
                    dataSubPlot(2).Position = [0.15 0.325 0.81 0.18];
                    dataSubPlot(1).Position = [0.15 0.09  0.81 0.18];
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
        'TickDir', 'out', ...
        'box', 'on');  
end