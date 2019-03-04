function label = ClassifyX(input, parameters)

%% All implementation should be inside the function.
Weights_IH = parameters(1);
Weights_IH = Weights_IH(1);
Weights_IH = cell2mat(Weights_IH);
Weights_HH = parameters(2);
Weights_HH = Weights_HH(1);
Weights_HH = cell2mat(Weights_HH);
Weights_HO = parameters(3);
Weights_HO = Weights_HO(1);
Weights_HO = cell2mat(Weights_HO);

nHidden1 = 20; %number of hidden nodes of first hidden layer
nHidden2 = 20; %number of hidden nodes of second hidden layer
nOutputs = 5; %number of output nodes of output layer
Bias_H1 = ones(nHidden1,1); %generate the bias for the first hidden layer
Bias_H2 = ones(nHidden2,1); %generate the bias for the second hidden layer
Bias_0 = ones(nOutputs,1); %generate the bias for the output layer
%% Testing    
Outputs = [];
Training_Store = [];
clc; disp('Now Classifying...'); pause(2);
timer_start=tic;
N = length(input);
beta = 0.1;
Hidden_Output1 = 1/1+exp(-beta*(Weights_IH * transpose(input))+Bias_H1);
Hidden_Output2 = 1/1+exp(-beta*(Weights_HH * Hidden_Output1) + Bias_H2);
Output = 1/1+exp(-beta*(Weights_HO * Hidden_Output2)+ Bias_0);

clc; disp('Classification Done!!');
disp(strcat('Execution Time: [',num2str(toc(timer_start)),'] seconds.')); pause(2); %display total time taken to classify
New_Output = transpose(Output);
 for k=1:N
          [mV1,mV2] = max(New_Output(k,:),[],2);
          train_Predictions(k,:) = [mV2,mV1]; 
 end

label = train_Predictions(:,1); %Column matrix with all labels

end