clear;
clc;

% First calculate privacy budget for each composition method 
% using cal_privacy_for_mnist.py (python code), which saves per iteration budgets in mat files
% Then, run this code which will load the saved mat files  

addpath(genpath('.'));
%%

whichComp = 1; % 0 for MA, 1 for SC

% choose either sigma=2 or sigma=1 for different values of totEps
sigma = 2;
%sigma = 1; 

% seednummat = 2;
% seednummat = linspace(1, 10, 10);
seednummat = 1:10; 
% seednummat = [6, 7, 8, 9, 10]; 
Hmat = [50,100];
Smat = [400,800,1600, 3200];
% Smat = 1600;
Itermat = 60000; % percent of the entire data


for seednum = seednummat
    rng(seednum, 'twister');

    load mnist_all.mat; % the whole dataset, 60,000 datapoints
    % data-preprocessing as binary variables
    traindata = 1*(traindata>0);
    testdata = 1*(testdata>0);

    [numdims,ntrain] = size(traindata); traindata = +(traindata>=rand(numdims,ntrain));
    [numdims,ntest] = size(testdata); testdata = +(testdata>=rand(numdims,ntest));
    
    selected_datapoints = randperm(10000);
    testdata_seed  = testdata(:, selected_datapoints(1:100));

    for K=Hmat
        for maxit=1:length(Itermat)
            for S = Smat
                
                opts.maxit = round(Itermat(maxit)/S); %  Itermat(maxit) means how many data you see
                %numIter = opts.maxit*(K+numdims+5);
                %numIter = opts.maxit; 
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
%                     [S, K, iterEps, opts.maxit]
                    
                    
                else % whichComp==1 % Strong Composition
                    filename_to_load = ['privacy_budget_SC_S=' num2str(S) '_K=' num2str(K) '_sigma=' num2str(sigma) '.mat'];
                    
                    b = load(filename_to_load);
                    iterEps = b.budget_SC(1);
                    iterDel = b.budget_SC(2);
                    totEps = b.budget_SC(3);
                    
%                     [S, K, iterEps, opts.maxit]
                    
                end
                    

                opts.iterEps = iterEps;
                opts.iterDel = iterDel;

                filename = ['pri_mn_seed=' num2str(seednum) '_nH=' num2str(K) '_N=' num2str(Itermat(maxit)) '_S=' num2str(S) 'eps=' num2str(totEps) '_whichComp=' num2str(whichComp) '_sigma=' num2str(sigma) '.mat']
                
               
                opts.mcsamples = 1; opts.interval = 1; opts.plotNow = 1;
                opts.batchsize = S; 
                opts.sigma = sigma; 
                
                result_sbnvb = sbn_vb_priv(traindata,testdata_seed,K,opts);

                save(filename, 'result_sbnvb');

                
            end
        end
    end
end
