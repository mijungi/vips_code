clear all; 
close all; 
clc;
% randn('state',100); rand('state',100);
addpath(genpath('.'));

seednum = 1; 
rng(seednum, 'twister');

%% load mnist data

load mnist_all.mat; % the whole dataset, 60,000 datapoints 

selected_datapoints = randperm(10000);
testdata = testdata(:, selected_datapoints(1:100));

[numdims,ntrain] = size(traindata); traindata = +(traindata>=rand(numdims,ntrain));
[numdims,ntest] = size(testdata); testdata = +(testdata>=rand(numdims,ntest));

%% SBN
% rand('state',200);

K = 50; 

% sbn + vb
opts.maxit = 300; opts.mcsamples = 1; opts.interval = 1; opts.plotNow = 1;
opts.batchsize = 100; 

numIter = opts.maxit*(K+numdims+5);
whichComp = 2; %  whichComp: 0 (linear), 1 (adv), 2 (zCDP) 
totEps = 1; % total privacy budget
totDel = 0.001; % tolerance
sampRate = opts.batchsize/ntrain;
[iterEps, iterDel]= cal_amp_eps(numIter, whichComp, totEps, totDel, sampRate);

opts.iterEps = iterEps; % per-iteration budget calculated by cal_amp_eps
opts.iterDel = iterDel;

result_sbnvb = sbn_vb_nonpriv(traindata,testdata,K,opts);

%%

figure(1); plot([result_sbnvb.TrainAcc', result_sbnvb.TestAcc']);
figure(2); dispims(result_sbnvb.W,28,28);
