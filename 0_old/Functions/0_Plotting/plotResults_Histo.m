function plotResults_Histo(GroundTruth, ID_Detected)

% --- Data input
A = histcounts(ID_Detected);
if(GroundTruth.Exists)
    A = [A; histcounts(GroundTruth.IstClusterID)];
end

% --- Plotten
figure;
bar(A);

end