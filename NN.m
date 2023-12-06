% to run the code correctly you need to change the 5th line to your own working directory
% Set-up
clear; clc; 
% Please change to your own working directory
working_dir = 'C:\Users\39427\Desktop\BMEG3330\BMEG3330 Project 2 Sample Code';
cd(working_dir); % Change the working directory to where your Sample Code folder is stored
FileName = 'Group3';
DataFolder = ['training_data/' FileName];
load([DataFolder '/EEGData.mat']);
fs = SampleRate;
% For each trials, extract 5 intervals of 1-second for training samples
% 0.5-1.5s, 1.5-2.5s, 2.5-3.5s 3.5-4.5s 4.5-5.5s 
% discard first and last 0.5s 

duration = 5; % Sample data only consist of 4s-long trials, Change to 5 for your own data
AllData = zeros(2, fs, 0); % (chan x time x trials) , Change channel number to 2 for your own data
for i = 1:length(Stimulus_Type)
    start = Stimulus_TimeMark(2*i-1);
    stop = Stimulus_TimeMark(2*i);
    data = EEGData(1 + round(start*fs) : round(stop*fs), :); 
    for j = 1:duration
        interval = round(1 + (j - 0.5) * fs) : round((j + 0.5) * fs);
        tmp = data(interval,:);
        tmp = tmp'; % convert to (chan x time) dimension
        AllData(:,:,end+1) = tmp;
    end
end
AllLabel = repelem(Stimulus_Type, 5); % For each trial, 3 segments of 1s data is extracted, Change to 5 for your own data
% Save data
save([DataFolder '/AllData.mat'], 'AllData', 'AllLabel');
% intialize eeglab
[~, EEG,~,~] = eeglab;
% Visual inspection to remove unwanted data
load([DataFolder '/AllData.mat']);
% Use EEGlab to review and reject data
EEG = pop_importdata('dataformat','array','nbchan',0,'data','AllData','setname','AllData','srate',fs,'pnts',0,'xmin',0);
EEG = eeg_checkset(EEG);
pop_eegplot(EEG,1,1,0);
waitfor(findobj('parent', gcf, 'string', 'UPDATE MARKS'), 'userdata'); % click UPDATE MARKS
EEG = eeg_checkset(EEG);
reject_trials = EEG.reject.rejmanual; 
% remove marked trials (both data and label) and save as new data
EEG = pop_rejepoch(EEG, reject_trials);

if isempty(reject_trials)
    label = AllLabel;
else
    label = AllLabel(~reject_trials); 
end
AllData = EEG.data; 
save([DataFolder '/rAllData.mat'], 'AllData', 'label')

% Feature extraction procedure (IMPORTANT: You can work on your own)
load([DataFolder '/rAllData.mat'])
AllData = double(AllData);
EEG = pop_importdata('dataformat', 'array', 'nbchan', 0, 'data', 'AllData', 'setname', 'AllData', 'srate', fs, 'pnts', 0, 'xmin', 0);

% Choose what frequency band you want to extract
% Example 
% FilterBand = {[8 13]}; Extract alpha power only
% FilterBand = {[8,30]}; Extract power from 8-30Hz (one feature)
% FilterBand = {[8 13], [13,30]};Extract alpha and beta power(two features)

FilterBand = {[8 13], [13,30]}; 

SavedFilter = {}; 
X = [];
for i = 1:length(FilterBand)
    [EEGFiltered , ~ , SavedFilter{i}] = pop_eegfiltnew(EEG, 'locutoff',FilterBand{i}(1),'hicutoff',FilterBand{i}(2));
    EEGFiltered = permute(EEGFiltered.data,[2,1,3]);
    Power = sum(EEGFiltered.^2,1)/size(EEGFiltered,1);
    Power = squeeze(Power); 
    X = [X Power']; % Horizontal concatenation to append features of different band 
end
% NumTrials = size(X,1);
% Save all filter obtained from EEGLAB
save([DataFolder '/SavedFilter.mat'], 'SavedFilter')

% Save feature matrix
save([DataFolder '/X.mat'], 'X', 'label');

% Model training (IMPORTANT)
load([DataFolder '/X.mat'])
NumTrials = length(label); 
NumClasses = 4; 

% k-fold cross validation procedures
% 1. Split data into k-blocks
% 2. Use k-1 blocks of data to train the model, and use the results to make prediction of the remaining block, to obtain accuracy of the model
% 3. Repeat the above process with differnet combinations, to obtain average accuracy of the model
% 4. Example, k=4, Train [1 2 3] block, predict [4], then Train [1 2 4], predict [3], Then Train [2,3,4], predict[1] etc 

k = 4; 
BlockLength = floor(NumTrials/k); 
index = randperm(NumTrials); 
% generate random blocks of data
data_block = {}; 
TrainAccuracy = zeros(1,k); % k accuracy value 
TestAccuracy = zeros(1,k);
% for i = 1:k
% 
%     TmpIndex = (1:BlockLength) + (i-1)*BlockLength ;
%     TmpIndex = TmpIndex(TmpIndex<=NumTrials); % Prevent index out of range
%     TrainData = X(index(setdiff(1:NumTrials, TmpIndex)),:);
%     TrainLabel = label(index(setdiff(1:NumTrials, TmpIndex))); 
%     TestData = X(index(TmpIndex),:);
%     TestLabel = label(index(TmpIndex));
% 
%     % Train the model
%     % SVM classifier
%     Mdl = fitcecoc(TrainData, TrainLabel, 'Learners', templateSVM('KernelFunction', 'polynomial', 'PolynomialOrder', 2, 'KernelScale', 'auto'), 'Coding', 'onevsall', 'ClassNames', [0 1 2 3]);
% 
%     % Calculate accuracy using TrainingLabel
%     PredictedLabel = predict(Mdl, TrainData);
%     TrainAccuracy(i) = sum(TrainLabel == PredictedLabel)./length(PredictedLabel);
% 
%     % Predict TestData using Trained Models
%     PredictedLabel = predict(Mdl,TestData);
%     TestAccuracy(i) = sum(TestLabel == PredictedLabel)./length(PredictedLabel);
% 
% end
% 
% mean(TestAccuracy)


% Model Performance Evaluation and save trained model

NumClasses = 4;
TrainData = X;
TrainLabel = label;
TrainLabel = categorical(TrainLabel);
numFeatures = size(TrainData,2);
% Import the necessary toolbox
import deepLearningToolbox.*

% Define the neural network architecture
layers = [
    featureInputLayer(numFeatures)
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(32, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(NumClasses, 'Name', 'output')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classification')
];

% Set the training options
options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'InitialLearnRate', 0.1, ...
    'Plots', 'training-progress' ...
    );

% Train the neural network

net = trainNetwork(TrainData, TrainLabel, layers, options);

% Make predictions using the trained network
PredictedLabel = classify(net, TrainData);

% Calculate the prediction matrix
PredictionMatrix = zeros(NumClasses);
for i = 1:length(PredictedLabel)
    PredictionMatrix(double(PredictedLabel(i)) + 1, label(i) + 1) = PredictionMatrix(double(PredictedLabel(i)) + 1, label(i) + 1) + 1;
end

PredictionMatrix;

% Calculate precision and recall
Precision = diag(PredictionMatrix) ./ sum(PredictionMatrix, 2);
Recall = diag(PredictionMatrix)' ./ sum(PredictionMatrix, 1);

% Save the trained model
save([DataFolder '/Trained_Net.mat'], 'net');







                                     








    
    


