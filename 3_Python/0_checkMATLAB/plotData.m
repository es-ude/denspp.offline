close all;
clear all;
clc;

load("Data.mat")
if(0)
    DataIn = Train_Input;
    DataOut = Train_Output;
else
    DataIn = YPredIn;
    DataOut = YPredOut;
end

idx = 1;
X = DataIn(idx,:);
Y = DataOut(idx,:);

h = figure(1);
p1 = plot(X, 'k', 'Linewidth', 1);
hold on;    grid on;
p2 = plot(Y, 'r', 'Linewidth', 1);
t = title("Index: 1");

p1.YDataSource = "X";
p2.YDataSource = "Y";

for idx = 2:1:size(DataIn,1)
    X = DataIn(idx,:);
    Y = DataOut(idx,:);
    title(strcat("Index: ", num2str(idx)));

    refreshdata;
    drawnow;

    pause(0.75);
end