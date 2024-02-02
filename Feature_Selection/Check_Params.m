%% This code is part of Feature_Selection toolbox. The toolbox is part of the following article. 
%% Please refer to it for more details about this code and cite it if you used this code.

%% Ghasemzadeh, H., Hillman, R. E., & Mehta, D. D. (2023). "Toward Generalizable Machine Learning Models in Speech, Language, and Hearing Sciences: Estimating Sample Size and Reducing Overfitting"
%% Journal of Speech, Language, and Hearing Research (JSLHR) https://doi.org/10.1044/2023_JSLHR-23-00273

%% By Hamzeh Ghasemzadeh
%% Email: hghasemzadeh@mgh.harvard.edu


function Params = Check_Params(Default_Params, Params)

Fields = fieldnames(Default_Params);
for Field_Counter = 1:numel(Fields)
    Current_Field = Fields(Field_Counter);
    Current_Field = Current_Field{1};
    if isfield(Params,Current_Field)==0
       Params.(Current_Field) = Default_Params.(Current_Field); 
    end
end
end
