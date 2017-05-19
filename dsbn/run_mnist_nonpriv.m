clear all; close all; clc;
% randn('state',100); rand('state',100);
addpath(genpath('.'));

%% load mnist data
load mnist_all.mat; % the whole dataset, 60,000 datapoints 

seednummat = linspace(1, 10, 10);
Hmat = [50, 100];
% Smat = [10, 50, 100, 200];
Smat = [50, 100, 200];
Itermat = [0.25].*60000; % percent of the entire data
% Itermat = [0.1, 0.2, 0.5].*60000; % percent of the entire data

[numdims,ntrain] = size(traindata); traindata = +(traindata>=rand(numdims,ntrain));
[numdims,ntest] = size(testdata); testdata = +(testdata>=rand(numdims,ntest));

for seednum = seednummat
    rng(seednum, 'twister');
    
    selected_datapoints = randperm(10000);
    testdata = testdata(:, selected_datapoints(1:100));

    for K=Hmat
        for maxit=1:length(Itermat)
            for S = Smat
                filename = ['nonpri_mn_seed=' num2str(seednum) '_nH=' num2str(K) '_N=' num2str(Itermat(maxit)) '_S=' num2str(S)]
                
                opts.maxit = Itermat(maxit)/S; %  Itermat(maxit) means how many data you see
                opts.mcsamples = 1; opts.interval = 1; opts.plotNow = 1;
                opts.batchsize = S; 
                
                result_sbnvb = sbn_vb_nonpriv(traindata,testdata,K,opts);

                save(filename, 'result_sbnvb');

            end
        end
    end
end
