%% This code is part of Feature_Selection toolbox. The toolbox is part of the following article. 
%% Please refer to it for more details about this code and cite it if you used this code.

%% Ghasemzadeh, H., Hillman, R. E., & Mehta, D. D. (2023). "Toward Generalizable Machine Learning Models in Speech, Language, and Hearing Sciences: Estimating Sample Size and Reducing Overfitting"
%% Journal of Speech, Language, and Hearing Research (JSLHR) https://doi.org/10.1044/2023_JSLHR-23-00273

%% By Hamzeh Ghasemzadeh
%% Email: hghasemzadeh@mgh.harvard.edu

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This function implements forward feature selection with validation accuracy of training-validation-testing cross-validation as the optimization cost function.

function [Testing_Accuracy, Selected_Indexes, History] = TrainValidationTest_FFS(Features, Labels, Selection_Params)

if nargin<3
    Selection_Params = [];
end
Default_Params.Fold_No = 10;
Default_Params.Selected_FeatureNo = 2;
Default_Params.Test_Ratio = 0.15;

Selection_Params = Check_Params(Default_Params, Selection_Params);
Fold_No = Selection_Params.Fold_No;
Selected_FeatureNo = Selection_Params.Selected_FeatureNo;
Test_Ratio = Selection_Params.Test_Ratio;

Outer_Partition = cvpartition(Labels,'Holdout',Test_Ratio);
TrainingValidation_Indexes = training(Outer_Partition);
Testing_Indexes = test(Outer_Partition);
Test_Labels = Labels(Testing_Indexes);
Test_Features = Features(Testing_Indexes, :);
TrainingValidation_Labels = Labels(TrainingValidation_Indexes);
TrainingValidation_Features = Features(TrainingValidation_Indexes, :);

Inner_Partitoins = cvpartition(TrainingValidation_Labels,'KFold',Fold_No);
[~, History] = sequentialfs(@Evaluation_Function, TrainingValidation_Features, TrainingValidation_Labels,'cv', Inner_Partitoins, 'nfeatures', Selected_FeatureNo);
Selected_Indexes = [];
Testing_Accuracy = zeros(1,Selected_FeatureNo);
for Feat_Counter = 1:Selected_FeatureNo
    Selected_Indexes = union(Selected_Indexes, find(History.In(Feat_Counter,:)),'stable');
    Score = Evaluation_Function(TrainingValidation_Features(:, Selected_Indexes), TrainingValidation_Labels, Test_Features(:, Selected_Indexes), Test_Labels);
    Testing_Accuracy(Feat_Counter) = 1-(Score/length(Test_Labels));
end

History.Selection_Params = Selection_Params;

end



