%% This code is part of Feature_Selection toolbox. The toolbox is part of the following article. 
%% Please refer to it for more details about this code and cite it if you used this code.

%% Ghasemzadeh, H., Hillman, R. E., & Mehta, D. D. (2023). "Toward Generalizable Machine Learning Models in Speech, Language, and Hearing Sciences: Estimating Sample Size and Reducing Overfitting"
%% Journal of Speech, Language, and Hearing Research (JSLHR) https://doi.org/10.1044/2023_JSLHR-23-00273

%% By Hamzeh Ghasemzadeh
%% Email: hghasemzadeh@mgh.harvard.edu

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This function performs consensus between different estimates of the best feature sets (one per outer fold).

function [Selected_Indexes, Stats] = Consense_NestedSelectedFeatures(Selection_Consistency, Feature_No)

Current_Selection_Consistency = Selection_Consistency;
Selected_Indexes = zeros(Feature_No,1);
Stats = zeros(Feature_No,1);
for Selection_Counter = 1:Feature_No
    [Hist,Bins] = Unique_Hist(Current_Selection_Consistency(:,Selection_Counter)');
    [~, Max_Index] = max(Hist);
    Selected_Indexes(Selection_Counter) = Bins(Max_Index);
    Temp = ismember(Selection_Consistency(:,1:Selection_Counter),Selected_Indexes(1:Selection_Counter));
    Temp = sum(Temp, 2);
    Flag = Temp==Selection_Counter;
    Stats(Selection_Counter) = sum(Flag);
    Current_Selection_Consistency = Selection_Consistency(Flag,:);
end
end

function  [Hist, Bins] = Unique_Hist(Data)
Unique = unique(Data);
[Hist, Bins] = hist(Data, Unique);
end