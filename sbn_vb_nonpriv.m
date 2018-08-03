function result = sbn_vb_nonpriv(Vtrain_tot,Vtest,K,opts)
%% Bayesian Inference for Sigmoid Belief Network via variational inference
%% With privacy preservation
% Zhe Gan (zhe.gan@duke.edu) wrote the non-private version of code (on
% Sep.1.2014)
% Mijung Park added the privatisation step in his codd (on Nov 16, 2016)
%
% V = sigmoid(W*H+c), H = sigmoid(b)
% Input:
%       Vtrain: p*ntrain training data
%       Vtest:  p*ntest  test     data
%       K:      number of latent hidden units
%       opts:   parameters of variational inference
% Output:
%       result: inferred matrix information


[p,ntrain_tot] = size(Vtrain_tot); [~,ntest] = size(Vtest);
ntrain = opts.batchsize;
% epsilon = opts.iterEps;
% delta = opts.iterDel;

%% initialize W and H
c = 0.1*randn(p,1); b = 0.1*randn(K,1);
prob = 1./(1+exp(-b));
Htrain = +(repmat(prob,1,ntrain)>rand(K,ntrain));
Htest = +(repmat(prob,1,ntest)>rand(K,ntest));

W = 0.1*randn(p,K); EWW = W.*W;
invgammaW = ones(p,K); xi = 1/2*ones(p,K); phi = ones(K,1); w = 1/2;

% TPBN prior: if you want to impose strong shrinkage, please do not infer
% phi, and set phi to be a small number, say, 1e-4. i.e.
% phi = 1e-4;
% invgammaW = 1/phi*ones(p,K); xi = 1/2*1/phi*ones(p,K);
% % phi = 1e-4*ones(K,1);

%% initialize vb parameters
iter = 0;
TrainAcc = zeros(1,opts.maxit); TestAcc = zeros(1,opts.maxit);
TotalTime = zeros(1,opts.maxit);
% TrainLogProb = zeros(1,opts.maxit);
% TestLogProb = zeros(1,opts.maxit);
% 
% num = opts.mcsamples;

tau0 = 1024;
kappa = 0.7;
boost = ntrain_tot/ntrain;

%% variational inference
tic;
while (iter < opts.maxit)
    
    iter = iter + 1;
    
    rhot = (tau0+iter).^(-kappa);
    
    idx_chosen = randperm(ntrain_tot);
    Vtrain = Vtrain_tot(:, idx_chosen(1:ntrain));
    
    %% noised up average vTrain
%     avg_vTrain = sum(Vtrain, 2)/ntrain;
    %     noisedup_avg_vTrain =
    
%     sen = sqrt(p)/ntrain_tot;
%     noise = Gaussian_noise(epsilon, delta, p, sen);
%     noisedup_avg_vTrain = avg_vTrain + noise;
%     noisedup_avg_vTrain(noisedup_avg_vTrain>1)=1;
%     noisedup_avg_vTrain(noisedup_avg_vTrain<0)=0;
%     noisedup_sum_vTrain = ntrain*noisedup_avg_vTrain;
    
    %% 1. update gamma0
    % this is an approximation, we ignore the uncertainty in W & H.
    X = bsxfun(@plus,W*Htrain,c);
    gamma0Train = 1/2./(X+realmin).*tanh(X/2+realmin);
    
    X = bsxfun(@plus,W*Htest,c);
    gamma0Test = 1/2./(X+realmin).*tanh(X/2+realmin);
    
    % 2. update H: sequential update
    % below we take the expectation of H directly. A better way is to
    % calculate the mean by taking several samples instead.
    res = W*Htrain;
    kset=randperm(K);
    for k = kset
        res = res - W(:,k)*Htrain(k,:);
        mat1 = bsxfun(@plus,res,c);
        vec1 = sum(bsxfun(@times,Vtrain-0.5-gamma0Train.*mat1,W(:,k))); % 1*n
        vec2 = sum(bsxfun(@times,gamma0Train,EWW(:,k)))/2; % 1*n
        logz = vec1 - vec2 + b(k); % 1*n
        probz = 1./(1+exp(-logz)); % 1*n
        Htrain(k,:) = probz;
        res = res + W(:,k)*Htrain(k,:);
    end;
    
    res = W*Htest;
    kset=randperm(K);
    for k = kset
        res = res - W(:,k)*Htest(k,:);
        mat1 = bsxfun(@plus,res,c);
        vec1 = sum(bsxfun(@times,Vtest-0.5-gamma0Test.*mat1,W(:,k))); % 1*n
        vec2 = sum(bsxfun(@times,gamma0Test,EWW(:,k)))/2; % 1*n
        logz = vec1 - vec2 + b(k); % 1*n
        probz = 1./(1+exp(-logz)); % 1*n
        Htest(k,:) = probz;
        res = res + W(:,k)*Htest(k,:);
    end;
    
    %% 3. update W
    SigmaW = 1./(boost*gamma0Train*Htrain'+invgammaW);
    jset=randperm(p);
    for j = jset
        Hgam = bsxfun(@times,Htrain,gamma0Train(j,:));
        HH = Hgam*Htrain'+diag(sum(Hgam.*(1-Htrain),2));

        invSigmaW = diag(invgammaW(j,:)) + boost*HH;

        MuW = invSigmaW\(boost*(sum(bsxfun(@times,Htrain,Vtrain(j,:)-0.5-c(j)*gamma0Train(j,:)),2)));
        W(j,:) = MuW';
    end;

    % convert Mu and Sigma to natural parameters, then shift
    na_par1 = (1./SigmaW).*W;
    na_par2 = 1./SigmaW; % J times N
    na_par_W = [na_par1; na_par2];
    if iter ==1
        na_par_old_W = na_par_W;
    else
        na_par_W = na_par_old_W*(1-rhot) + rhot*na_par_W;
        na_par_old_W = na_par_W;
    end
    
    % convert back
    SigmaW = 1./na_par_old_W(p+1:end, :);
    W = SigmaW.*na_par_old_W(1:p, :);
    
    EWW = W.^2 + SigmaW;
    
    %% 4. update c
    sigmaC = 1./(boost*sum(gamma0Train,2)+1);
    c = sigmaC.*(boost*sum(Vtrain-0.5-gamma0Train.*(W*Htrain),2));
    
    % convert Mu and Sigma to natural parameters, then shift
    na_par1 = (1./sigmaC).*c;
    na_par2 = 1./sigmaC; % J times N
    na_par_C = [na_par1; na_par2];
    if iter ==1
        na_par_old_C = na_par_C;
    else
        na_par_C = na_par_old_C * (1-rhot) + rhot *na_par_C;
        na_par_old_C = na_par_C;
    end
    
    % convert back
    sigmaC = 1./na_par_old_C(p+1:end, :);
    c= sigmaC.*na_par_old_C(1:p, :);
    
    
    %% 5. update b
    gamma1 = 1/2./(b+realmin).*tanh(b/2+realmin);
    sigmaB = 1./(boost*ntrain*gamma1+1);
    b = sigmaB.*(boost*sum(Htrain-0.5,2));
    % convert Mu and Sigma to natural parameters, then shift
    na_par1 = (1./sigmaB).*b;
    na_par2 = 1./sigmaB; % J times N
    na_par = [na_par1; na_par2];
    if iter ==1
        na_par_old = na_par;
    else
        na_par = na_par_old * (1-rhot) + rhot *na_par;
        na_par_old = na_par;
    end
    
    % convert back
    sigmaB = 1./na_par_old(K+1:end, :);
    b = sigmaB.*na_par_old(1:K, :);
    
    
    %% update of the TPBN parameter
    % (1). update gammaW
    a_GIG = 2*xi; b_GIG = EWW; sqrtab = sqrt(a_GIG.*b_GIG);
    gammaW = (sqrt(b_GIG).*besselk(1,sqrtab))./(sqrt(a_GIG).*besselk(0,sqrtab));
    invgammaW = (sqrt(a_GIG).*besselk(1,sqrtab))./(sqrt(b_GIG).*besselk(0,sqrtab));
    
    % update of the TPBN parameter
    % (2). update xi
    a_xi = 1; b_xi = bsxfun(@plus,gammaW,phi');
    xi = a_xi./b_xi;
    
    % (3). update phi
    a_phi = 0.5+0.5*p; b_phi = w + sum(xi)';
    phi = a_phi./b_phi;
    % (4). update w
    w = (0.5+0.5*K)/(1+sum(phi));
    
    %% 6. reconstruct the images
    sampleHtrain = Htrain >=rand(K,ntrain);
    sampleHtest = Htest >=rand(K,ntest);
    
    X = bsxfun(@plus,W*sampleHtrain,c); % p*n
    prob = 1./(1+exp(-X));
    VtrainRecons = (prob>0.5);
    
    X = bsxfun(@plus,W*sampleHtest,c); % p*n
    prob = 1./(1+exp(-X));
    VtestRecons = (prob>0.5);
    
    TrainAcc(iter) = sum(sum(VtrainRecons==Vtrain))/p/ntrain;
    TestAcc(iter) = sum(sum(VtestRecons==Vtest))/p/ntest;
    
%     % 7. calculate the lower bound
%     % The marginal likelihood can be evaluated by using the
%     % "functionEvaluation" file in the folder "support" using AIS.
%     totalP0 = zeros(1,num);
%     for i = 1:num
%         Hsamp = Htrain>=rand(K,ntrain);
%         mat1 = bsxfun(@plus,W*Hsamp,c);
%         totalP0(i) = sum(sum(mat1.*Vtrain-log(1+exp(mat1)))); 
%     end;
%     trainP0 = mean(totalP0)/ntrain;
%     mat1 = bsxfun(@times,Htrain,b);
%     trainP1 = sum(sum(mat1))-ntrain*sum(log(1+exp(b))); trainP1 = trainP1/ntrain;    
%     trainQ1 = sum(sum(Htrain.*log(Htrain+realmin)+(1-Htrain).*log(1-Htrain+realmin))); trainQ1 = trainQ1/ntrain;
%     TrainLogProb(iter) = trainP0+trainP1-trainQ1;
% 
%     totalP0 = zeros(1,num);
%     for i = 1:num
%         Hsamp = Htest>=rand(K,ntest);
%         mat1 = bsxfun(@plus,W*Hsamp,c);
%         totalP0(i) = sum(sum(mat1.*Vtest-log(1+exp(mat1)))); 
%     end;
%     testP0 = mean(totalP0)/ntest;
%     mat1 = bsxfun(@times,Htest,b);
%     testP1 = sum(sum(mat1))-ntest*sum(log(1+exp(b))); testP1 = testP1/ntest;    
%     testQ1 = sum(sum(Htest.*log(Htest+realmin)+(1-Htest).*log(1-Htest+realmin))); testQ1 = testQ1/ntest;
%     TestLogProb(iter) = testP0+testP1-testQ1;

    
    TotalTime(iter) = toc;
    
    if mod(iter,opts.interval)==0
        disp(['Iteration: ' num2str(iter) ' Acc: ' num2str(TrainAcc(iter)) ' ' num2str(TestAcc(iter))]);
        %         disp(['Iteration: ' num2str(iter) ' Acc: ' num2str(TrainAcc(iter)) ' ' num2str(TestAcc(iter))...
        %             ' LogProb: ' num2str(TrainLogProb(iter))  ' ' num2str(TestLogProb(iter))...
        %              ' Totally spend ' num2str(TotalTime(iter))]);
        
        if  opts.plotNow == 1
            %             index = randperm(ntrain);
            %             figure(1);
            %             dispims(VtrainRecons(:,index(1:100)),28,28); title('Reconstruction');
            figure(2);
            subplot(1,2,1); imagesc(W); colorbar; title('W');
            subplot(1,2,2); imagesc(Htrain); colorbar; title('Htrain');
            figure(3);
            dispims(W,28,28); title('dictionaries');
            drawnow;
        end;
    end
end;

result.W = W; result.b = b; result.c = c;
result.Htrain = Htrain; result.Htest = Htest;
result.gamma0Train = gamma0Train;
result.gamma0Test = gamma0Test;
result.gamma1 = gamma1;
result.TrainAcc = TrainAcc;
result.TestAcc = TestAcc;
result.TotalTime = TotalTime;
% result.TrainLogProb = TrainLogProb;
% result.TestLogProb = TestLogProb;

