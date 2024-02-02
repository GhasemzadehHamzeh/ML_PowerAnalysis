%% This function compute the requires sample size for achieving a target C2,2 (the probability that both selected features are correct) for given values of:
%% D_0 (Cohens'D of the two selected features [if they are not equal, aproximate it by averaging]), m_0 (number of extracted feature), and CI_0 (target C2,2).
%% 0.4<=D_0<=1, 10<=m_0<=40,

%% This code is part of the following article. Please refer to it for more details about this code and cite it if you used this code.

%% Ghasemzadeh, H., Hillman, R. E., & Mehta, D. D. (2023). "Toward Generalizable Machine Learning Models in Speech, Language, and Hearing Sciences: Estimating Sample Size and Reducing Overfitting"
%% Journal of Speech, Language, and Hearing Research (JSLHR) https://doi.org/10.1044/2023_JSLHR-23-00273

%% By Hamzeh Ghasemzadeh
%% Email: hghasemzadeh@mgh.harvard.edu

function Sample_Size = Compute_RecommendedSampleSize(m_0, D_0, CI_0)
%%% Tables are from the appendix of the paper.
%%% Each table hold the values of C22 for a given value of m (extracted features)
%%% Columns are for different values of D (Cohen's D)
%%% Rows are for different number of pairs

Confidence{1} = [17.7	27.9	40.3	50.2	60.8	68.6	75.1
                38.2	51.7	66.9	78.3	85.6	90.9	94.2
                52.7	69.3	81.4	90.3	94.7	97.2	98.4
                63.4	79.7	90.1	95.7	98.7	99.5	99.8
                72.9	88.3	95.9	98.5	99.6	99.9	100
                79.6	90.5	96.6	99.1	99.8	100 	nan
                84.7	94.5	98.6	99.7	99.9	nan	    nan
                88.1	96.1	99.1	99.8	100	    nan	    nan
                90.3	97.3	99.6	100	    nan	    nan	    nan
                92.6	98.6	99.9	nan	    nan	    nan	    nan ];

Confidence{2} = [9.5	17.3	27	    37.4	48.7	59.4	65.5
                23.8	40.1	55.7	68	    79	    85.8	90.6
                39.5	59.5	75	    86.3	92.6	96.5	98
                51.3	71.5	85.8	93.5	96	    98.6	99.4
                63.3	83.2	92.5	96.8	99.2	99.7	99.8
                73.4	88.4	96.9	99.1	99.7	100	    100
                79	    92	    97.5	99.4	99.8	nan	    nan
                84.1	94.9	99	    99.8	100	    nan	    nan
                88.1	96.8	99.2	99.9	nan	    nan	    nan
                90.3	97.6	99.7	100 	nan	    nan	    nan];

Confidence{3} = [6.3	11.9	19.7	31.5	40.9	50.7	59.7
                19.3	35.3	52.3	67.5	77.6	85.4	90.1
                32.6	53.6	70.6	83.7	90.5	94.7	97.7
                48.4	69.8	84.5	92.3	96.6	98.8	99.4
                56.8	77.5	90.6	96.3	98.7	99.5	99.9
                66.1	84	    94.2	97.9	99.3	99.8	100
                75.8	89.8	96.5	99.4	99.9	100	    nan
                81.2	94.1	98.7	99.8	100	    nan	    nan
                84.8	95.7	98.9	100 	nan	    nan	    nan
                86.9	96.5	99.5	nan	    nan	    nan	    nan];

Confidence{4} = [4.8	10.3	16.6	26.5	38.2	48.2	57.5
                15.1	31.7	46.3	60.9	72.8	81.2	87.8
                29.2	50.3	67.5	81	    89.8	94.4	97.6
                41.8	66.8	82.4	91.4	95.4	98.4	99.4
                53.3	74.1	89.3	95.4	98.3	99.5	100
                63.1	81.6	93	    98	    99.4	99.8	nan
                70.8	89.1	95.4	98.8	99.6	99.9	nan
                76.1	91.2	97.8	99.4	99.9	100 	nan
                82.9	94.9	98.8	99.7	100	    nan	    nan
                86.8	97.3	99.6	99.9	nan	    nan	    nan];


SampleSize_Table = repmat((50:50:500)',1,7);
Table_D = repmat(0.4:0.1:1,10,1);
Table_m = 10:10:40;

if m_0<10 || m_0>40
    error('Input parameters are out of range! The valid ranges are: 10<=m_0<=40, 0.4<=D_0<=1');
end
if D_0<0.4 || D_0>1
    error('Input parameters are out of range! The valid ranges are: 10<=m_0<=40, 0.4<=D_0<=1');
end


TargetTable_Index = floor(m_0/10);
Target_Table = Confidence{TargetTable_Index};
% ft = 'linearinterp';
ft = 'thinplateinterp';

% Fit model to data.

[xData, yData, zData] = prepareSurfaceData( Table_D, Target_Table, SampleSize_Table);
[fitresult, gof] = fit([xData, yData], zData, ft, 'Normalize', 'on' );
Sample_Size = fitresult(D_0, CI_0);

if mod(m_0,10)>0
    Next_Table = Confidence{TargetTable_Index+1};

    % Fit model to data.

    [xData, yData, zData] = prepareSurfaceData( Table_D, Next_Table, SampleSize_Table);
    [fitresult, gof] = fit( [xData, yData], zData, ft, 'Normalize', 'on' );
    Next_SampleSize = fitresult(D_0, CI_0);
    X_Values = Table_m(TargetTable_Index:TargetTable_Index+1);
    Y_Values = [Sample_Size Next_SampleSize];
    Slope = diff(Y_Values)/diff(X_Values);
    Sample_Size = Slope*(m_0-X_Values(1))+Y_Values(1);
end

if Sample_Size<=10
    warning('Out of bound result!')
    Sample_Size = nan;
end
