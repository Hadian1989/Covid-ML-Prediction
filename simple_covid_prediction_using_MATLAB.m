clc
clear
close all
%%
%I = xlsread('data_samples_without consequence-remove wrong and nan label.csv');
%I = xlsread('data_samples_remove transportation field.csv');
% I = xlsread('data_samples_without consequence-remove wrong label.csv');
I = xlsread('data_samples_two class-1_2_.csv');
% I = xlsread('data_samples_without consequence-remove wrong and nan label.csv');

missing_sum = sum(ismissing(I),2);
% O_sum=sum(isoutlier(I(:,2:end)),2);
% O_sum2=sum( isoutlier(I),2);
% O= isoutlier(I)
% idx = find(O_sum2>=2);
idx = find(sum(missing_sum,2)>=2);
I(idx,:)=[];
IN= I(:,1:end-1);
label=I(:,end);
%IN = mapminmax(IN, 0 , 1);
%%
trainData= IN(1:100,:);
%trainData= IN(1:300,:);
trainLabel= label(1:100);
%trainLabel= label(1:300);

testData= IN(101:end,:);
%testData= IN(301:end,:);
testLabel= label(101:end,:);
%% Training:
Model_knn = fitcknn(trainData,trainLabel);
t = templateSVM('Standardize', true);
Model_svm = fitcecoc(trainData, trainLabel, 'Learners', t);
Model_tree = fitctree(trainData,trainLabel);
%Model_nb = fitcnb(trainData,trainLabel);
%% Validation:
Y_hat_knn = predict(Model_knn, testData);
Y_hat_svm = predict(Model_svm, testData);
% Y_hat_nb = predict(Model_nb, testData);
Y_hat_tree = predict(Model_tree, testData);
acc_knn= sum(Y_hat_knn == testLabel) / numel(testLabel)*100
acc_svm= sum(Y_hat_svm == testLabel) / numel(testLabel)*100
% acc_nb= sum(Y_hat_nb == testLabel) / numel(testLabel)*100
acc_tree= sum(Y_hat_tree == testLabel) / numel(testLabel)*100

