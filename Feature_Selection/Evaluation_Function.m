%% This code is part of Feature_Selection toolbox. The toolbox is part of the following article. 
%% Please refer to it for more details about this code and cite it if you used this code.

%% Ghasemzadeh, H., Hillman, R. E., & Mehta, D. D. (2023). "Toward Generalizable Machine Learning Models in Speech, Language, and Hearing Sciences: Estimating Sample Size and Reducing Overfitting"
%% Journal of Speech, Language, and Hearing Research (JSLHR) https://doi.org/10.1044/2023_JSLHR-23-00273

%% By Hamzeh Ghasemzadeh
%% Email: hghasemzadeh@mgh.harvard.edu

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This function implements the cost function of forward feature selection. 
%% The optimization process minimizes the cost function. The current implementation uses 1-accuracy (error rate) as the cost function. 
%% You may replace it with other cost functions depending on the application (e.g., balanced accuracy, AUC, etc.). 
%% Also, the current implementation uses logistic regression as classifier. If you want to use a different classifier you can update it here.

function Score = Evaluation_Function(Train_Features, Train_Labels, Test_Features, Test_Labels)

Model = fitglm(Train_Features, Train_Labels, 'Distribution', 'binomial', 'link', 'logit');  %% logistic regression 
Soft_Labels = predict(Model, Test_Features);
Soft_Labels = Soft_Labels-0.5;
Predicted_Labels = Soft_Labels>0;
CP = classperf(Test_Labels, Predicted_Labels);
Score = CP.ErrorRate*length(Test_Labels);  %% The optimization cost function 
end
