%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ll, pChoice, value, choiceValue, choiceProbs] = f_fitRL_rw(params,data,start_values)

% This function runs the RL model to the data structure below using the
% parameters provided

% data structure:
% 1  trial, 2 chosen (stimulus), 3 reward 
    
%%% parameters for the 3 paramater RL model
lr = params(1);
itemp = params(2);

%%% find out how many stimuli are being chosen between
nStim = length(unique(data(:, 2)));

%%% initialize the model values we will keep
%value of stimuli
value = zeros(size(data, 1), nStim);
% Choice probability of each option
pChoice = zeros(size(data, 1), nStim);
% Choice values
choiceValue = zeros(size(data, 1), nStim);
% Choice prob of chosen option
choiceProbs = zeros(size(data, 1), nStim);

%%% initialize likelihood, that measures model fit
ll = 0;

%%% if you are passing values over from previous fitted trials
if nargin > 2
    value(1,:) = start_values;
end

%trial loop
for trial = 1 : size(data, 1)
    
    %%% softmax to derive choice probabilities from values
    pChoice(trial, :) = exp(itemp*value(trial, :))./sum(exp(itemp*value(trial, :)));
    
    %%% all options are presented
    stimOptions = [1,2,3];
    
    % figure out what was chosen
    chosenStim = data(trial, 2);
    unChosen = stimOptions(find(stimOptions ~= chosenStim));
    unChosenStim1 = unChosen(:,1);
    unChosenStim2 = unChosen(:,2);
    
    %%% Save values and choice probs of options ordered by chosen and
    %%% Unchosen for plotting of performance
    choiceValue(trial, 1) = value(trial, chosenStim);
    choiceValue(trial, 2) = value(trial, unChosenStim1);
    choiceValue(trial, 3) = value(trial, unChosenStim2);
    
    choiceProbs(trial, 1) = pChoice(trial, chosenStim);
    choiceProbs(trial, 2) = pChoice(trial, unChosenStim1);
    choiceProbs(trial, 3) = pChoice(trial, unChosenStim2);

    %%% accumulate model fit statistics
    ll = ll - log(pChoice(trial, chosenStim));
    
    %%% update values using delta learning rule
    value(trial+1, :) = value(trial, :); % set previous trial value
    
    %%% Do the actual reinforcement learning stuff: v(t) = v(t-1) + lr(reward - v(t-1)) 
    value(trial+1, chosenStim) = value(trial, chosenStim) + lr*(data(trial, 3) - ...
        value(trial, chosenStim));
    
end % end trial loop
