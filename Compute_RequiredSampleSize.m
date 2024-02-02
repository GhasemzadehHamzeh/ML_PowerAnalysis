%% The required sample size for getting a significant outcome from machine learning, alpha = 0.05, Beta = 0.2

%% This code is part of the following article. Please refer to it for more details about this code and cite it if you used this code.

%% Ghasemzadeh, H., Hillman, R. E., & Mehta, D. D. (2023). "Toward Generalizable Machine Learning Models in Speech, Language, and Hearing Sciences: Estimating Sample Size and Reducing Overfitting"
%% Journal of Speech, Language, and Hearing Research (JSLHR) https://doi.org/10.1044/2023_JSLHR-23-00273

%% By Hamzeh Ghasemzadeh
%% Email: hghasemzadeh@mgh.harvard.edu

function Sample_Size = Compute_RequiredSampleSize(D, m, l)
%% D is the effect size of discriminative features
%% m is the number of extracted features
%% l is the number of selected features

a =  39.37 -6.718*l + 0.263*m;
b = -1.985 -0.023*l + 0.001*m;
c = -0.886 +1.507*l - 0.015*m;
Sample_Size = a.*D.^b+c;