clear all; close all; clc;

% First calculate privacy budget for each composition method 
% using cal_privacy_for_mnist.py (python code), which saves per iteration budgets in mat files
% Then, run this code which will load the saved mat files  

addpath(genpath('.'));

%% load mnist data
load mnist_all.mat; % the whole dataset, 60,000 datapoints 

whichComp = 0; % 0 for MA, 1 for SC

% choose either sigma=2 or sigma=1 for different values of totEps
% sigma = 2;
sigma = 1; 

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
                
%                 [iterEps, iterDel]= cal_amp_eps(numIter, whichComp, totEps, totDel, sampRate);
                if whichComp==0 % MA
                    % load budget file
                    filename_to_load = ['privacy_budget_MA_S=' num2str(S) '_K=' num2str(K) '_sigma=' num2str(sigma) '.mat'];
                    
                    b = load(filename_to_load);
                    iterEps = b.budget_MA(1);
                    iterDel = b.budget_MA(2);
                    totEps = b.budget_MA(3);
                    
                else % whichComp==1 % Strong Composition
                    filename_to_load = ['privacy_budget_SC_S=' num2str(S) '_K=' num2str(K) '_sigma=' num2str(sigma) '.mat'];
                    
                    b = load(filename_to_load);
                    iterEps = b.budget_SC(1);
                    iterDel = b.budget_SC(2);
                    totEps = b.budget_SC(3);
                end
                    
                    
                
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
