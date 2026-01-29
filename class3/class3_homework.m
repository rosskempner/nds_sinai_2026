% class3work.m

% load in the data
class3data = readmatrix('class3data_homework.txt');

%% datafile information
% Inputs 
% data_file: in the format: 
%       Column
%       1 - Subject (1-n)
%       2 - Schedule (you can ignore this for now)
%       3 - Session Number (1-8: there are 8 sessions)
%       4 - Trial Number (1-300: each session is 300 trials)
%       5 - Choice (0-2: there are three buttons)
%       6 - Reward (0-1: reward/no reward)
%       7 - ReactionTime (0-30000ms: subjects had 30s to respond)

% The choice numbers in the datafile are 0 - 2 (needs to be 1-3 to map to columns)
class3data(:,5)=class3data(:,5)+1;

% You want to run the analysis on trials, choices and rewards
% for simplicity select these three columns
schedule_data = class3data(:,4:6);

% set the max evaluations for fminsearch 
options = optimset('MaxFunEvals', 1000, 'TolFun', 0.00001);

%%% set some initial values for this fitting
% this is in order of LR and then IT [LR IT];
params = [0.5 5];
    
% %create matrices before the FOR loop for efficiency (Pre-assigning makes MATLAB happy)
LR=nan(1,1); IT=nan(1,1); LL=nan(1,1);

%%% optimize the parameters to the session using fminsearchbnd
% The search for parameters is bounded so that MATLAB doesn't return
% values for the parameters that are unreasonable.
%[mparams, lla] = fminsearchbnd(@(params) fitRL_rw(params, schedule_data), params,[0 0],[1 30], options);

[mparams, lla] = fminsearch(@(params) fitRL_rw(params, schedule_data), params, options);

% Save the fitted parameter estimates into the matrices for outputting
LR(:,1)=mparams(1); IT(:,1)=mparams(2); 

%%% Run the RL model using fitted parameters 
[ll, pChoice, pStimvalue, choiceValue, choiceProbs] = f_fitRL_rw(mparams, schedule_data);

% Save the log likelihoods from the model that used the best parameters
LL(:,1)=ll;

% Compute AIC and BIC for comparison to other models
AIC(1,1) = 2*LL + 2*length(mparams);
BIC(1,1) = (log(length(schedule_data)) * length(mparams)) + (2*LL); % log(n)k - 2LL

%%  Single session trial by trial plots

% set length of the data. Check the length of your data. 
data_length = 300;

figure;
subplot(2,2,1);
plot(pStimvalue);
title('Stimulus Values')
box off
axis([1 data_length 0 1])
xlabel('Trials');
ylabel('Modeled Value');

%%% Choice prob of each option
subplot(2,2,2);
plot(pChoice);
title('Choice Probabilities')    
box off
axis([1 data_length 0 1])
xlabel('Trials');
ylabel('Choice Probability');

%%% Choice prob of chosen option
subplot(2,2,3);
plot(choiceProbs(:, 1));
title('Choice Probability of Chosen Option')
box off
axis([1 data_length 0 1])
xlabel('Trials');
ylabel('Choice Probability');

% Define correct on the basis of what has been learned about each option
% If you average this across your data it should be close to the actual
% behavioral choice accuracy (I bet you aren't still reading)
subplot(2,2,4);
correctChoice = choiceProbs(:,1) - max(choiceProbs(:,2:3),[],2) > 0;
plot(smooth(correctChoice,5,'moving'));
title('Choice of Best Model Option')
box off
axis([1 data_length 0 1])
xlabel('Trials');
ylabel('Correct Choice (0 or 1)');

%% ClASS 3 HOMEWORK %% 
% Download the class3data_homework.txt file. 
% This file is in the same format as the file above as follows 

% Inputs 
% data_file: in the format: 
%       Column
%       1 - Subject (1-n)
%       2 - Schedule (you can ignore this for now)
%       3 - Session Number (1-8: there are 8 sessions)
%       4 - Trial Number (1-300: each session is 300 trials)
%       5 - Choice (0-2: there are three buttons)
%       6 - Reward (0-1: reward/no reward)
%       7 - ReactionTime (0-30000ms: subjects had 30s to respond)

% Now there are 4 subjects (column 1 - 1-4) and each subject performed the task in 8
% separate sessions (column 3 - 1-8). 

% Using all that you have learned from the classes about loops and structures 
% as well as the code that I have given you here, your assignment is to fit the 
% two parameter model to each session from 4 monkeys and make a swarm or violin plot of the learning 
% rates and inverse parameters temperatures from the 8 sessions for each subject. 
% Each plot should have one violin per subject. 

% HINT: A simple approach to this problem is to use a FOR loop and maybe two together. 

% Please send the file with the two plots to me peter.rudebeck@mssm.edu as a single .pdf
% before the start of the next class on 3/27/2025.
% When you send the single .pdf file please put your last name at the start
% of the file name (e.g. Rudebeck_NDS_class3_homework.pdf).

