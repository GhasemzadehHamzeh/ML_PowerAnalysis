%% This code is part of Feature_Selection toolbox. The toolbox is part of the following article.
%% Please refer to it for more details about this code and cite it if you used this code.

%% Ghasemzadeh, H., Hillman, R. E., & Mehta, D. D. (2023). "Toward Generalizable Machine Learning Models in Speech, Language, and Hearing Sciences: Estimating Sample Size and Reducing Overfitting"
%% Journal of Speech, Language, and Hearing Research (JSLHR) https://doi.org/10.1044/2023_JSLHR-23-00273

%% By Hamzeh Ghasemzadeh
%% Email: hghasemzadeh@mgh.harvard.edu

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This file demonstrates the application of forward feature selection with four different cross-validation methods described in the above-referenced article.
% % The code estimates the distribution of H0 (accuracy of non-discriminative features) for different cross-validations.
% % To estimate the distribution of Ha, you need to set the BWClass_Distance variable to your desired effect size.

clc;
clear;
warning off;

%%% Simulation parametrs
Feat_No = 20; %% Dimensionality of the feature space (m)
Pair_No = 50; %% Number of pairs (N)
BWClass_Distance = 0; %% Cohen.s D of discriminative features. Set this to "0" for H0 and to a non-zero value for Ha scenarios.
Experiment_No = 1000; %% Number of Monte Carlo simulations (this number was set to 5,000 in the paper)

Fold_No = 10;
SelectedFeature_No = 2; %% Number of selected features (l)

SingleHoldout_SelectionParams.Selected_FeatureNo = SelectedFeature_No;
SingleHoldout_SelectionParams.Test_Ratio = 0.3;


KFold_SelectionParams.Fold_No = Fold_No;
KFold_SelectionParams.Selected_FeatureNo = SelectedFeature_No;

TrainValidationTest_SelectionParams.Fold_No = Fold_No;
TrainValidationTest_SelectionParams.Selected_FeatureNo = SelectedFeature_No;
TrainValidationTest_SelectionParams.Test_Ratio = 0.15;

Nested_SelectionParams.Fold_No = Fold_No;
Nested_SelectionParams.Selected_FeatureNo = SelectedFeature_No;

SingleHoldout_TestingAccuracy = zeros(Experiment_No, SelectedFeature_No);
KFold_TestingAccuracy = zeros(Experiment_No, SelectedFeature_No);
TrainValidationTest_TestingAccuracy = zeros(Experiment_No, SelectedFeature_No);
Nested_TestingAccuracy = zeros(Experiment_No, SelectedFeature_No);


parfor Experiment_Counter = 1:Experiment_No
    Experiment_Counter
    Class1 = randn(Pair_No,Feat_No);
    Class2 = randn(Pair_No,Feat_No);

    %% If BWClass_Distance>0, only the second and next to last features are discriminative.
    %% The goal of feature selection is to find these features.
    Class2(:,2) = Class2(:,2) + BWClass_Distance;
    Class2(:,end-1) = Class2(:,end-1) + BWClass_Distance;  

    Features = [Class1; Class2];
    Labels = [zeros(Pair_No,1); ones(Pair_No,1)];

    [SingleHoldout_TestingAccuracy(Experiment_Counter,:), SingleHoldout_SelectedIndexes] = SingleHoldout_FFS(Features, Labels, SingleHoldout_SelectionParams);
    [KFold_TestingAccuracy(Experiment_Counter,:), KFold_SelectedIndexes] = KFold_FFS(Features, Labels, KFold_SelectionParams);
    [TrainValidationTest_TestingAccuracy(Experiment_Counter,:), TrainValidationTest_SelectedIndexes] = TrainValidationTest_FFS(Features, Labels, TrainValidationTest_SelectionParams);
    [Nested_TestingAccuracy(Experiment_Counter,:), Nested_SelectedIndexes] = NestedKFold_FFS(Features, Labels, Nested_SelectionParams);
end
Hist_Data(:,1) = SingleHoldout_TestingAccuracy(:,2);
Hist_Data(:,2) = KFold_TestingAccuracy(:,2);
Hist_Data(:,3) = TrainValidationTest_TestingAccuracy(:,2);
Hist_Data(:,4) = Nested_TestingAccuracy(:,2);

hist(Hist_Data, 50);
xlabel('Testing accuracy');
ylabel('Count');
legend('Single Holdout','10-fold','Train-Validation-Test','Nested 10-fold');
