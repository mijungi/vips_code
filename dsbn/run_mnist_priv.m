clear all; close all; clc;
% randn('state',100); rand('state',100);
addpath(genpath('.'));

%% load mnist data
load mnist_all.mat; % the whole dataset, 60,000 datapoints 

whichComp = 1;
totEps = 0.5;

seednummat = 1; 
%seednummat = linspace(1, 10, 10);
% seednummat = [6, 7, 8, 9, 10]; 
Hmat = [50, 100];
Smat = [50, 100, 200];
Itermat = [0.25].*60000; % percent of the entire data

[numdims,ntrain] = size(traindata); traindata = +(traindata>=rand(numdims,ntrain));
[numdims,ntest] = size(testdata); testdata = +(testdata>=rand(numdims,ntest));


for seednum = seednummat
    rng(seednum, 'twister');
    
    selected_datapoints = randperm(10000);
    testdata_seed  = testdata(:, selected_datapoints(1:100));

    for K=Hmat
        for maxit=1:length(Itermat)
            for S = Smat
                
                opts.maxit = Itermat(maxit)/S; %  Itermat(maxit) means how many data you see
                numIter = opts.maxit*(K+numdims+5);
                totDel = 0.0001;
                sampRate = S/ntrain;
                [iterEps, iterDel]= cal_amp_eps(numIter, whichComp, totEps, totDel, sampRate);
                
                opts.iterEps = iterEps;
                opts.iterDel = iterDel;

                filename = ['pri_mn_seed=' num2str(seednum) '_nH=' num2str(K) '_N=' num2str(Itermat(maxit)) '_S=' num2str(S) 'eps=' num2str(totEps) '_whichComp=' num2str(whichComp)]
                
               
                opts.mcsamples = 1; opts.interval = 1; opts.plotNow = 1;
                opts.batchsize = S; 
                
                result_sbnvb = sbn_vb_priv(traindata,testdata_seed,K,opts);

                save(filename, 'result_sbnvb');

            end
        end
    end
end