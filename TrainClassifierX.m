function parameters = TrainClassifierX(input, label)

%% All implementation should be inside the function.
% Name:Charis Georgiou
% CID:01600146
%---- Normalise Data ---
new_input = normalize(input);
z = length(new_input);
% %---- Plot Data ---
figure;
for i =1:64
    subplot(8,8,i)
    X = input(:,i+1);
    plot(z,label,z,X)
    title('Feature - Labels Correlation')
    xlabel(['Feature' num2str(i)])
    ylabel('Labels')
end
%% Configurations/Parameters
nInputs = 64; %number of input nodes of input layer
nHidden1 = 20; %number of hidden nodes of first hidden layer
nHidden2 = 20; %number of hidden nodes of second hidden layer
nOutputs = 5; %number of output nodes of output layer
learningrate = 1.85; %set learning rate
a = 1; %use this variable to generate weights between -1 and 1
b = -1; %use this variable to generate weights between -1 and 1
Weights_IH = (b-a).*rand(nHidden1,nInputs) + a; %generate set of weights from input to hidden layer connections
Weights_HH = (b-a).*rand(nHidden2,nHidden1) + a; %generate set of weights from hidden to hidden layer connections
Weights_HO = (b-a).*rand(nOutputs,nHidden2) + a; %generate set of weights from hidden to output layer connections
Bias_H1 = ones(nHidden1,1); %generate the bias for the first hidden layer
Bias_H2 = ones(nHidden2,1); %generate the bias for the second hidden layer
Bias_0 = ones(nOutputs,1); %generate the bias for the output layer

%% Training
%randomly shuffle the data
[numDataPts,temp] = size(new_input);
elems = randperm (numDataPts)';
    
Outputs = [];
Errors = [];
Training_Store = [];
Testing_Store = [];
Testing_Errors = [];
Testing_Outputs = [];
Outputs_Average = [];
Testing_Outputs_Average = [];
K = 10;
clc; disp('Now Training...'); pause(2);
timer_start=tic;
for fold = 1:K %10 folds of 2400 data rows each
     disp('==============================================================');
     disp(strcat('Fold Number: [',num2str(fold),']')); %Show current fold
     disp(strcat('Training [%',num2str((fold/10)*100),'] Completed')); %Show percentage of training completed
     disp('=============================================================='); 
     % Training And Testing Input/Output Selection 
        test = elems(((fold-1)*(numDataPts/10)+1:((fold-1)*(numDataPts/10))+(numDataPts/10)));
        train = setdiff(elems,test);
        train_Input = new_input(train,:); %21600 data rows for training
        new_train_Output = label(train,:);
        test_Input = new_input(test,:); %2400 data rows for testing
        new_test_Output = label(test,:);
        N = length(train_Input);
        Z = length(test_Input);
        train_Output=zeros(N,nOutputs);
        for k=1:N
          train_Output(k,new_train_Output(k))= 1; %format training output e.g class 3 is now 00100
        end
        test_Output=zeros(Z,nOutputs);
        for k=1:Z
          test_Output(k,new_test_Output(k))= 1;%format testing output
        end
        % Feedforward
        beta = 0.1; %Gain parameter of activation function
        for k = 1:N %for the training dataset
        Hidden_Output1 = [];
        for j=1:nHidden1
          Hidden_Output1(j,:)= act_function((Weights_IH(j,:)*transpose(train_Input(k,:))+Bias_H1(j)),beta); %Calculate the training outputs of the first hidden layer
        end
        Hidden_Output2 = [];
        for j=1:nHidden2
          Hidden_Output2(j,:)= act_function(((Weights_HH(j,:)* Hidden_Output1 + Bias_H2(j))),beta); %Calculate the training outputs of the second hidden layer
        end          
        Output = [];
        for j=1:nOutputs
          Output(j,:)= act_function(((Weights_HO(j,:)* Hidden_Output2)+ Bias_0(j)),beta); %calculate the final training outputs
        end  
       Output_new = transpose(Output);
       error = [];
       for j=1:nOutputs
          error(1,j) = train_Output(k,j) - Output_new(1,j); %prediction minus target
       end
        Errors(k,:) = error; %store training errors
        Outputs(k,:) = Output; %store training outputs
        % Backpropagation
        delta_HO = [];
        for j=1:nOutputs
            delta_HO(j) = ((train_Output(k,j) - Output_new(1,j)))* Output(j)*(1-Output(j)); %Error times gradient
        end 
        delta_HH = [];
        for j=1:nHidden2
            delta_HH(j,:)=(1-(Hidden_Output2(j) * Hidden_Output2(j)))*(transpose(Weights_HO(:,j))* transpose(delta_HO)); %Backpropagate error to the second hidden layer
        end 
        delta_IH = [];
        for j=1:nHidden1
            delta_IH(j,:)=(1-(Hidden_Output1(j) * Hidden_Output1(j)))*(transpose(Weights_HH(:,j)) * delta_HH); %Backpropagate error to first hidden layer
        end
        Delta_IH = learningrate * delta_IH * train_Input(k,:); 
        Weights_IH = Weights_IH + Delta_IH; % updating the weights from input to hidden layer 1   
        Delta_HH = learningrate * delta_HH * transpose(Hidden_Output1);
        Weights_HH = Weights_HH + Delta_HH; % updating the weights from hidden layer 1 to hidden layer 2 
        Delta_HO = learningrate * transpose(delta_HO) * transpose(Hidden_Output2);
        Weights_HO = Weights_HO + Delta_HO; % updating the weights from hidden layer 2 to output layer
        end  
        %% Testing
        for k = 1:Z %for the testing data set
        Testing_Hidden_Output1 = [];
        for j=1:nHidden1
          Testing_Hidden_Output1(j,:)= act_function((Weights_IH(j,:)*transpose(test_Input(k,:))+Bias_H1(j)),beta); %Calculate the testing outputs of the first hidden layer
        end
        Testing_Hidden_Output2 = [];
        for j=1:nHidden2
          Testing_Hidden_Output2(j,:)= act_function(((Weights_HH(j,:)* Testing_Hidden_Output1 + Bias_H2(j))),beta); %Calculate the testing outputs of the second hidden layer
        end          
        Testing_Output = [];
        for j=1:nOutputs
          Testing_Output(j,:)= act_function(((Weights_HO(j,:)* Testing_Hidden_Output2)+ Bias_0(j)),beta); %calculate the final training outputs
        end  
       Testing_Output_new = transpose(Testing_Output);
       Testing_error = [];
       for j=1:nOutputs
          Testing_error(1,j) = test_Output(k,j) - Testing_Output_new(1,j); %prediction minus target
       end
        Testing_Errors(k,:) = Testing_error; %Store testing errors 
        Testing_Outputs(k,:) = Testing_Output_new; %Store testing outputs
        % Backpropagation
        Testing_delta_HO = [];
        for j=1:nOutputs
            Testing_delta_HO(j) = ((test_Output(k,1) - Testing_Output_new(1,j)))* Testing_Output(j)*(1-Testing_Output(j));
        end 
        Testing_delta_HH = [];
        for j=1:nHidden2
            Testing_delta_HH(j,:)=(1-(Testing_Hidden_Output2(j) * Testing_Hidden_Output2(j)))*(transpose(Weights_HO(:,j))* transpose(Testing_delta_HO));
        end 
        Testing_delta_IH = [];
        for j=1:nHidden1
            Testing_delta_IH(j,:)=(1-(Testing_Hidden_Output1(j) * Testing_Hidden_Output1(j)))*(transpose(Weights_HH(:,j)) * Testing_delta_HH);
        end
        Testing_Delta_IH = learningrate * Testing_delta_IH * test_Input(k,:);
        Weights_IH = Weights_IH + Testing_Delta_IH; % updating the weights    
        Testing_Delta_HH = learningrate * Testing_delta_HH * transpose(Testing_Hidden_Output1);
        Weights_HH = Weights_HH + Testing_Delta_HH; % updating the weights
        Testing_Delta_HO = learningrate * transpose(Testing_delta_HO) * transpose(Testing_Hidden_Output2);
        Weights_HO = Weights_HO + Testing_Delta_HO; % updating the weights
        end  
        Total_Outputs{1,fold} = Outputs; %Store training outputs for all folds
        TT{1,fold} = Testing_Outputs; %Store testing outputs for all folds
end
clc; disp('Training Done!!');
disp(strcat('Execution Time: [',num2str(toc(timer_start)),'] seconds.')); pause(2); %display total time taken to train
A = cat(3,Total_Outputs{:});
B = cat(3,TT{:});
Outputs_Average = mean(A,3); %average of 10 training output matrices over 10 folds
Testing_Outputs_Average = mean(B,3); %average of 10 testing output matrices for all folds
 for k=1:N
          [mV1,mV2] = max(train_Output(k,:),[],2);
          train_Predictions(k,:) = [mV1,mV2]; %format training predictions e.g 00100 to class 3
 end
 for k = 1:Z
 [tV1,tV2] = max(test_Output(k,:),[],2);
 test_Predictions(k,:) = [tV1,tV2]; %format testing predictions
 end 

correct = 0;
Training_Accuracy = [];
y = transpose(1:1:N);
for k = 1:N %for the training dataset
  [maxVals,maxLocs] = max(Outputs_Average(k,:),[],2); %store the max output and the column which is found
  Training_Store(k,:) = [maxVals,maxLocs];
  if Training_Store(k,2) == train_Predictions(k,2) %if prediction matches output add one to correct
    correct = correct + 1; 
  end
  Training_Accuracy(k,1) = correct/y(k,1);
end        
training_Accuracy = (correct/N)*100 %calculate training accuracy (correct training outputs/number of training outputs)

correct = 0;
Testing_Accuracy = [];
y = transpose(1:1:Z);
for k = 1:Z
  [maxVals,maxLocs] = max(Testing_Outputs_Average(k,:),[],2);
  Testing_Store(k,:) = [maxVals,maxLocs];
  if Testing_Store(k,2) == test_Predictions(k,2)
    correct = correct + 1;
  end
  Testing_Accuracy(k,1) = correct/y(k,1);
end        
testing_Accuracy = (correct/Z)*100 %calculate testing accuracy (correct testing outputs/number of testing outputs)
Total_Labels = vertcat(train_Predictions(:,2),test_Predictions(:,2)); %Column matrix with all labels
Total_Predictions = vertcat(Training_Store(:,2),Testing_Store(:,2)); %Column matrix with all predictions from classifier
% Confusion Matrix
T11 = 0;
F12 = 0;
F13 = 0;
F14 = 0;
F15 = 0;
T22 = 0;
F21 = 0;
F23 = 0;
F24 = 0;
F25 = 0;
T33 = 0;
F31 = 0;
F32 = 0;
F34 = 0;
F35 = 0;
T44 = 0;
F41 = 0;
F42 = 0;
F43 = 0;
F45 = 0;
T55 = 0;
F51 = 0;
F52 = 0;
F53 = 0;
F54 = 0;
for k = 1:z
    if Total_Labels(k,1) ==1 && Total_Predictions(k,1) == 1 %if actual label is 1 and prediction is 1
        T11 = T11 + 1;
    elseif Total_Labels(k,1) == 1 && Total_Predictions(k,1) == 2 %if actual label is 1 and prediction is 2
        F12 = F12 + 1;
    elseif Total_Labels(k,1) == 1 && Total_Predictions(k,1) == 3 %if actual label is 1 and prediction is 3
        F13 = F13 + 1;
    elseif Total_Labels(k,1) == 1 && Total_Predictions(k,1) == 4 %if actual label is 1 and prediction is 4
        F14 = F14 + 1;
    elseif Total_Labels(k,1) == 1 && Total_Predictions(k,1) == 5 %if actual label is 1 and prediction is 5
        F15 = F15 + 1; 
    end
    if Total_Labels(k,1) == 2 && Total_Predictions(k,1) == 2 %if actual label is 2 and prediction is 2
        T22 = T22 + 1;
    elseif Total_Labels(k,1) == 2 && Total_Predictions(k,1) == 1 %if actual label is 2 and prediction is 1
        F21 = F21 + 1;
    elseif Total_Labels(k,1) == 2 && Total_Predictions(k,1) == 3 %if actual label is 2 and prediction is 3
        F23 = F23 + 1;
    elseif Total_Labels(k,1) == 2 && Total_Predictions(k,1) == 4 %if actual label is 2 and prediction is 4
        F24 = F24 + 1;
    elseif Total_Labels(k,1) == 2 && Total_Predictions(k,1) == 5 %if actual label is 2 and prediction is 5
        F25 = F25 + 1; 
    end
    if Total_Labels(k,1) == 3 && Total_Predictions(k,1) == 3 %if actual label is 3 and prediction is 3
        T33 = T33 + 1;
    elseif Total_Labels(k,1) == 3 && Total_Predictions(k,1) == 1 %if actual label is 3 and prediction is 1
        F31 = F31 + 1;
    elseif Total_Labels(k,1) == 3 && Total_Predictions(k,1) == 2 %if actual label is 3 and prediction is 2
        F32 = F32 + 1;
    elseif Total_Labels(k,1) == 3 && Total_Predictions(k,1) == 4 %if actual label is 3 and prediction is 4
        F34 = F34 + 1;
    elseif Total_Labels(k,1) == 3 && Total_Predictions(k,1) == 5 %if actual label is 3 and prediction is 5
        F35 = F35 + 1; 
    end
     if Total_Labels(k,1) == 4 && Total_Predictions(k,1) == 4 %if actual label is 4 and prediction is 4
        T44 = T44 + 1;
    elseif Total_Labels(k,1) == 4 && Total_Predictions(k,1) == 1 %if actual label is 4 and prediction is 1
        F41 = F41 + 1;
    elseif Total_Labels(k,1) == 4 && Total_Predictions(k,1) == 2 %if actual label is 4 and prediction is 2
        F42 = F42 + 1;
    elseif Total_Labels(k,1) == 4 && Total_Predictions(k,1) == 3 %if actual label is 4 and prediction is 3
        F43 = F43 + 1;
    elseif Total_Labels(k,1) == 4 && Total_Predictions(k,1) == 5 %if actual label is 4 and prediction is 5
        F45 = F45 + 1; 
    end
    if Total_Labels(k,1) == 5 && Total_Predictions(k,1) == 5 %if actual label is 5 and prediction is 5
        T55 = T55 + 1;
    elseif Total_Labels(k,1) == 5 && Total_Predictions(k,1) == 1 %if actual label is 5 and prediction is 1
        F51 = F51 + 1;
    elseif Total_Labels(k,1) == 5 && Total_Predictions(k,1) == 2 %if actual label is 5 and prediction is 2
        F52 = F52 + 1;
    elseif Total_Labels(k,1) == 5 && Total_Predictions(k,1) == 3 %if actual label is 5 and prediction is 3
        F53 = F53 + 1;
    elseif Total_Labels(k,1) == 5 && Total_Predictions(k,1) == 4 %if actual label is 5 and prediction is 4
        F54 = F54 + 1; 
     end
end
Confusion_Matrix = [T11 F12 F13 F14 F15;
                    F21 T22 F23 F24 F25;
                    F31 F32 T33 F34 F35;
                    F41 F42 F43 T44 F45;
                    F51 F52 F53 F54 T55;]
    
plot(Training_Accuracy,'LineWidth',2)
grid on
hold on
plot(Testing_Accuracy,'LineWidth',2)
xlabel('Data','FontSize',16)
ylabel('Accuracy %','FontSize',16)
title('Training and Testing Accuracy of Neural Network','FontSize',16)
legend({'Training','Validation'},'FontSize',16)

parameters = {Weights_IH,Weights_HH,Weights_HO};

end

function out=act_function(in,beta)
    out=1./(1+exp(-beta*in));
end