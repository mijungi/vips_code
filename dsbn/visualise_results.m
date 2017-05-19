%% visualise results
% Mijung wrote on Nov 21, 2016

clear;
clc;
%% (1) non-private version

% seednummat = linspace(1, 10, 10);
seednummat = [1, 2, 3, 4]; 
Hmat = [50, 100];
Smat = [10, 50, 100, 200];
% Smat = [100, 200];
Itermat = [0.1, 0.2, 0.5].*60000; % percent of the entire data

H50_TestAcc = zeros(length(Smat), length(Itermat), length(seednummat));
H100_TestAcc = zeros(length(Smat), length(Itermat), length(seednummat));

for seednum = seednummat
    for K=Hmat
        for maxit=1:length(Itermat)
            for S = Smat
                filename = ['results/nonpri_mn_seed=' num2str(seednum) '_nH=' num2str(K) '_N=' num2str(Itermat(maxit)) '_S=' num2str(S)];
%                 display(filename)
                load(filename)
                if K==50
                    if S==10
                        H50_TestAcc(1, maxit, seednum) = result_sbnvb.TestAcc(end);
                    elseif S==50
                        H50_TestAcc(2, maxit, seednum) = result_sbnvb.TestAcc(end);
                    elseif S==100
                        H50_TestAcc(3, maxit, seednum) = result_sbnvb.TestAcc(end);
                    else
                        H50_TestAcc(4, maxit, seednum) = result_sbnvb.TestAcc(end);
                    end
                else % K==100
                    if S==10
                        H100_TestAcc(1, maxit, seednum) = result_sbnvb.TestAcc(end);
                    elseif S==50
                        H100_TestAcc(2, maxit, seednum) = result_sbnvb.TestAcc(end);
                    elseif S==100
                        H100_TestAcc(3, maxit, seednum) = result_sbnvb.TestAcc(end);
                    else
                        H100_TestAcc(4, maxit, seednum) = result_sbnvb.TestAcc(end);
                    end
                end
%                 plot(result_sbnvb.TestAcc)
%                 pause;
            end
        end
    end
end
            
%%

mean_H50 = mean(H50_TestAcc, 3);
std_H50 = std(H50_TestAcc, 0, 3);


mean_H100 = mean(H100_TestAcc, 3);
std_H100 = std(H100_TestAcc, 0, 3);

[mean_H50 mean_H100]
[std_H50 std_H100]


H50_nonpriv = H50_TestAcc; 
H100_nonpriv = H100_TestAcc;


load mnist_all.mat
selected_datapoints = randperm(10000);
testdata = testdata(:, selected_datapoints(1:100));
Vtest = testdata; 

ntest = 100;
K = 100;
W = result_sbnvb.W;
c = result_sbnvb.c;
b = result_sbnvb.b;

prob = 1./(1+exp(-b));
Htest = +(repmat(prob,1,ntest)>rand(K,ntest));

X = bsxfun(@plus,W*Htest,c);
gamma0Test = 1/2./(X+realmin).*tanh(X/2+realmin);
    
    
res = W*Htest;
kset=randperm(K);
EWW = W.*W;

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

sampleHtest = Htest >=rand(K,ntest);
    

X = bsxfun(@plus,W*sampleHtest,c); % p*n

for i=1:100
    imagesc(reshape(X(:,i), 28, [])); axis image;
    pause(0.5);
end
% boxplot([reshape(H50_TestAcc(3,3,:), [], 1), reshape(H50_TestAcc(4,3,:), [], 1), reshape(H100_TestAcc(3,3,:), [], 1), reshape(H100_TestAcc(4,3,:), [], 1)])
% H100, S=100, 50% of training data, gave me the best result for the
% non-private version

%% (2) private version : Adv

% seednummat = linspace(1, 10, 10); 
seednummat = [1, 2, 3, 4, 5, 6];
% seednummat = [1, 2]
Hmat = [50, 100];
Smat = [10, 50, 100, 200];
Itermat = [0.1, 0.2, 0.5].*60000; % percent of the entire data

whichComp = 1;
totEps = 1;

H50_TestAcc = zeros(length(Smat), length(Itermat), length(seednummat));
H100_TestAcc = zeros(length(Smat), length(Itermat), length(seednummat));

for seednum = seednummat
    for K=Hmat
        for maxit=1:length(Itermat)
            for S = Smat
                filename = ['results/pri_mn_seed=' num2str(seednum) '_nH=' num2str(K) '_N=' num2str(Itermat(maxit)) '_S=' num2str(S) 'eps=' num2str(totEps) '_whichComp=' num2str(whichComp)]; 
%                 display(filename)
                load(filename)
                if K==50
                    if S==10
                        H50_TestAcc(1, maxit, seednum) = result_sbnvb.TestAcc(end);
                    elseif S==50
                        H50_TestAcc(2, maxit, seednum) = result_sbnvb.TestAcc(end);
                    elseif S==100
                        H50_TestAcc(3, maxit, seednum) = result_sbnvb.TestAcc(end);
                    else
                        H50_TestAcc(4, maxit, seednum) = result_sbnvb.TestAcc(end);
                    end
                else % K==100
                    if S==10
                        H100_TestAcc(1, maxit, seednum) = result_sbnvb.TestAcc(end);
                    elseif S==50
                        H100_TestAcc(2, maxit, seednum) = result_sbnvb.TestAcc(end);
                    elseif S==100
                        H100_TestAcc(3, maxit, seednum) = result_sbnvb.TestAcc(end);
                    else
                        H100_TestAcc(4, maxit, seednum) = result_sbnvb.TestAcc(end);
                    end
                end
%                 plot(result_sbnvb.TestAcc)
%                 pause;
            end
        end
    end
end

%%

mean_H50 = mean(H50_TestAcc, 3);
std_H50 = std(H50_TestAcc, 0, 3);


mean_H100 = mean(H100_TestAcc, 3);
std_H100 = std(H100_TestAcc, 0, 3);

[mean_H50 mean_H100]
[std_H50 std_H100]

H50_Adv = H50_TestAcc; 
H100_Adv = H100_TestAcc;

ntest = 100;
K = 100;
W = result_sbnvb.W;
c = result_sbnvb.c;
b = result_sbnvb.b;

prob = 1./(1+exp(-b));
Htest = +(repmat(prob,1,ntest)>rand(K,ntest));

X = bsxfun(@plus,W*Htest,c);
gamma0Test = 1/2./(X+realmin).*tanh(X/2+realmin);
    
    
res = W*Htest;
kset=randperm(K);
EWW = W.*W;

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

sampleHtest = Htest >=rand(K,ntest);
    

X = bsxfun(@plus,W*sampleHtest,c); % p*n

for i=1:100
    imagesc(reshape(X(:,i), 28, [])); axis image;
    pause(0.5);
end

% boxplot([reshape(H50_Adv(3,3,:), [], 1), reshape(H50_Adv(4,3,:), [], 1), reshape(H100_Adv(3,3,:), [], 1), reshape(H100_Adv(4,3,:), [], 1)])


%% (3) private version : zCDP

% seednummat = linspace(1, 10, 10); 
seednummat = [1, 2, 3, 4, 5, 6];
% seednummat = [1, 2];
Hmat = [50, 100];
Smat = [10, 50, 100, 200];
Itermat = [0.1, 0.2, 0.5].*60000; % percent of the entire data

whichComp = 2;
totEps = 1;

H50_TestAcc = zeros(length(Smat), length(Itermat), length(seednummat));
H100_TestAcc = zeros(length(Smat), length(Itermat), length(seednummat));

for seednum = seednummat
    for K=Hmat
        for maxit=1:length(Itermat)
            for S = Smat
                filename = ['results/pri_mn_seed=' num2str(seednum) '_nH=' num2str(K) '_N=' num2str(Itermat(maxit)) '_S=' num2str(S) 'eps=' num2str(totEps) '_whichComp=' num2str(whichComp)]; 
%                 display(filename)
                load(filename)
                if K==50
                    if S==10
                        H50_TestAcc(1, maxit, seednum) = result_sbnvb.TestAcc(end);
                    elseif S==50
                        H50_TestAcc(2, maxit, seednum) = result_sbnvb.TestAcc(end);
                    elseif S==100
                        H50_TestAcc(3, maxit, seednum) = result_sbnvb.TestAcc(end);
                    else
                        H50_TestAcc(4, maxit, seednum) = result_sbnvb.TestAcc(end);
                    end
                else % K==100
                    if S==10
                        H100_TestAcc(1, maxit, seednum) = result_sbnvb.TestAcc(end);
                    elseif S==50
                        H100_TestAcc(2, maxit, seednum) = result_sbnvb.TestAcc(end);
                    elseif S==100
                        H100_TestAcc(3, maxit, seednum) = result_sbnvb.TestAcc(end);
                    else
                        H100_TestAcc(4, maxit, seednum) = result_sbnvb.TestAcc(end);
                    end
                end
%                 plot(result_sbnvb.TestAcc)
%                 pause;
            end
        end
    end
end

%%

mean_H50 = mean(H50_TestAcc, 3);
std_H50 = std(H50_TestAcc, 0, 3);


mean_H100 = mean(H100_TestAcc, 3);
std_H100 = std(H100_TestAcc, 0, 3);

[mean_H50 mean_H100]
[std_H50 std_H100]

H50_zCDP = H50_TestAcc; 
H100_zCDP = H100_TestAcc;


ntest = 100;
K = 100;
W = result_sbnvb.W;
c = result_sbnvb.c;
b = result_sbnvb.b;

prob = 1./(1+exp(-b));
Htest = +(repmat(prob,1,ntest)>rand(K,ntest));

X = bsxfun(@plus,W*Htest,c);
gamma0Test = 1/2./(X+realmin).*tanh(X/2+realmin);
    
    
res = W*Htest;
kset=randperm(K);
EWW = W.*W;

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

sampleHtest = Htest >=rand(K,ntest);
    

X = bsxfun(@plus,W*sampleHtest,c); % p*n

for i=1:100
    imagesc(reshape(X(:,i), 28, [])); axis image;
    pause(0.5);
end

prob = 1./(1+exp(-X));
VtestRecons = (prob>0.5);
p = 28*28;
err = sum(sum(VtestRecons==Vtest))/p/ntest

%%

% first plot for H=50

figure(1); 
subplot(1,3,1);
boxplot([reshape(H50_Adv(3,3,:), [], 1), reshape(H50_Adv(4,3,:), [], 1)]);
ylabel('prediction accuracy');
set(gca, 'ylim', [0.4, 1]);
subplot(1,3,2);
boxplot([reshape(H50_zCDP(3,3,:), [], 1), reshape(H50_zCDP(4,3,:), [], 1)]);
set(gca, 'ylim', [0.4, 1]);
title('H=50, S \in (100, 200), epsilon=1'); 
subplot(1,3,3);
    boxplot([reshape(H50_nonpriv(3,3,:), [], 1), reshape(H50_nonpriv(4,3,:), [], 1)])
set(gca, 'ylim', [0.4, 1]); 

figure(2); 
subplot(1,3,1);
boxplot([reshape(H100_Adv(3,3,:), [], 1), reshape(H100_Adv(4,3,:), [], 1)]);
set(gca, 'ylim', [0.4, 1]);
ylabel('prediction accuracy');
subplot(1,3,2);
boxplot([reshape(H100_zCDP(3,3,:), [], 1), reshape(H100_zCDP(4,3,:), [], 1)]);
set(gca, 'ylim', [0.4, 1]);
title('H=100, S \in (100, 200), epsilon=1'); 
subplot(1,3,3);
    boxplot([reshape(H100_nonpriv(3,3,:), [], 1), reshape(H100_nonpriv(4,3,:), [], 1)])
set(gca, 'ylim', [0.4, 1]); 




% second plot for H=100

% reshape(H100_Adv(3,3,:), [], 1), reshape(H100_Adv(4,3,:), [], 1)


%  dispims(result_sbnvb.W,28,28);




%%

% Htest = result_sbnvb.Htest;
% ntest = 100;
% K = 100;
% W = result_sbnvb.W;
% c = result_sbnvb.c;
% 
% sampleHtest = Htest >=rand(K,ntest);
%     
% 
%     X = bsxfun(@plus,W*sampleHtest,c); % p*n
%     prob = 1./(1+exp(-X));
%     VtestRecons = (prob>0.5);






