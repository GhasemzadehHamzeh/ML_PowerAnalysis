%% This code is part of Feature_Selection toolbox. The toolbox is part of the following article. 
%% Please refer to it for more details about this code and cite it if you used this code.

%% Ghasemzadeh, H., Hillman, R. E., & Mehta, D. D. (2023). "Toward Generalizable Machine Learning Models in Speech, Language, and Hearing Sciences: Estimating Sample Size and Reducing Overfitting"
%% Journal of Speech, Language, and Hearing Research (JSLHR) https://doi.org/10.1044/2023_JSLHR-23-00273

%% By Hamzeh Ghasemzadeh
%% Email: hghasemzadeh@mgh.harvard.edu

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This function implements forward feature selection with validation accuracy of nested k-fold cross-validation as the optimization cost function.

function [Testing_Accuracy, Selected_Indexes, History] = NestedKFold_FFS(Features, Labels, Selection_Params)

if nargin<3
    Selection_Params = [];
end
Default_Params.Fold_No = 10;
Default_Params.Selected_FeatureNo = 2;
Selection_Params = Check_Params(Default_Params, Selection_Params);
Fold_No = Selection_Params.Fold_No;
Selected_FeatureNo = Selection_Params.Selected_FeatureNo;

Outer_Partitoins = cvpartition(Labels,'KFold',Fold_No);
Folds_SelectedIndexes = zeros(Fold_No, Selected_FeatureNo);
Testing_Accuracy = zeros(Fold_No, Selected_FeatureNo);
for Fold_Counter = 1:Fold_No
    TrainingValidation_Indexes = find(training(Outer_Partitoins,Fold_Counter));
    TrainingValidation_Features = Features(TrainingValidation_Indexes,:);
    TrainingValidation_Labels = Labels(TrainingValidation_Indexes);

    Inner_Partitions = cvpartition(TrainingValidation_Labels,'KFold',Fold_No);
    [~, History] = sequentialfs(@Evaluation_Function, TrainingValidation_Features, TrainingValidation_Labels,'cv', Inner_Partitions, 'nfeatures', Selected_FeatureNo);   

    Test_Indexes = find(test(Outer_Partitoins,Fold_Counter));
    Test_Features = Features(Test_Indexes,:);
    Test_Labels = Labels(Test_Indexes);
    Selected_Indexes = [];
    for Feat_Counter = 1:Selected_FeatureNo
        Selected_Indexes = union(Selected_Indexes, find(History.In(Feat_Counter,:)),'stable');
        Score = Evaluation_Function(TrainingValidation_Features(:, Selected_Indexes), TrainingValidation_Labels, Test_Features(:, Selected_Indexes), Test_Labels);
        Testing_Accuracy(Fold_Counter, Feat_Counter) = 1-(Score/length(Test_Labels));
    end
    Folds_SelectedIndexes(Fold_Counter, :) = Selected_Indexes;
end

History.Fold_SelectedIndexes = Folds_SelectedIndexes;
History.Fold_TestingAccuracy = Testing_Accuracy;
History.Selection_Params = Selection_Params;

[Selected_Indexes, Stats] = Consense_NestedSelectedFeatures(Folds_SelectedIndexes, Selected_FeatureNo);
Testing_Accuracy = mean(Testing_Accuracy);
end



