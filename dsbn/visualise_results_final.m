%% visualise results
% Mijung wrote on Jan 16, 2018

clear;clc;

addpath(genpath('.'));
%% (1) non-private version

%seednummat = linspace(1, 10, 10);
seednummat = [1,2,3,4];
Hmat = [50, 100];
% Smat = [50, 100, 200];
Smat = [20, 400];
Itermat = [0.25].*60000; % percent of the entire data

H50_TestAcc = zeros(length(Smat), length(Itermat), length(seednummat));
H100_TestAcc = zeros(length(Smat), length(Itermat), length(seednummat));

for seednum = seednummat
    for K=Hmat
        for maxit=1:length(Itermat)
            for S = Smat
                filename = ['nonpri_mn_seed=' num2str(seednum) '_nH=' num2str(K) '_N=' num2str(Itermat(maxit)) '_S=' num2str(S)];
%                 display(filename)
                load(filename)
                if K==50
                    if S==20
                        H50_TestAcc(1, maxit, seednum) = result_sbnvb.TestAcc(end);
                    elseif S==50
                        H50_TestAcc(2, maxit, seednum) = result_sbnvb.TestAcc(end);
                    elseif S==100
                        H50_TestAcc(3, maxit, seednum) = result_sbnvb.TestAcc(end);
                    else
                        H50_TestAcc(4, maxit, seednum) = result_sbnvb.TestAcc(end);
                    end
                else % K==100
                    if S==20
                        H100_TestAcc(1, maxit, seednum) = result_sbnvb.TestAcc(end);
                    elseif S==50
                        H100_TestAcc(2, maxit, seednum) = result_sbnvb.TestAcc(end);
                    elseif S==100
                        H100_TestAcc(3, maxit, seednum) = result_sbnvb.TestAcc(end);
                    else
                        H100_TestAcc(4, maxit, seednum) = result_sbnvb.TestAcc(end);
                    end
                end % K=200
%                 plot(result_sbnvb.TestAcc)
%                 pause;
            end
        end
    end
end
            
%% plotting for non-private method

H50_nonpriv = squeeze(H50_TestAcc)';
H50_nonpriv =  H50_nonpriv(:,[1,4]);
figure(1);
subplot(1,3,3);
boxplot(H50_nonpriv);
set(gca, 'ylim', [0.4, 1]);
 title('non-priv');
% set(gca, 'xtick', Smat);

H100_nonpriv = squeeze(H100_TestAcc)';
H100_nonpriv =  H100_nonpriv(:,[1,4]);
figure(2);
subplot(1,3,3);
boxplot(H100_nonpriv);
set(gca, 'ylim', [0.4, 1]); 
 title('non-priv');

figure(3);
W = result_sbnvb.W;
dispims(W,28,28); title('non-priv');

%% (2) private version : Strong composition 

% seednummat = linspace(1, 10, 10); 
seednummat = [1,2,3,4]; 
Hmat = [50, 100];
% Smat = [50, 100, 200];
Smat = [20, 400]; 
Itermat = [0.25].*60000; % percent of the entire data

% For and whichComp = 0 (MA) and 

whichComp = 1 % (strong)
sigma = 1

%% this was with the wrong way to use MA and Strong composition
%================================= sigma=2 ========================

% ========= H = 50 model ===========
% (1) for S = 50, 
%       when sigma = 2, totEps= 1.1862

% (2) for S = 100,
%       when sigma = 2, totEps = 1.7005

% (3) for S = 200,
%       when sigma = 2, totEps = 2.4494


% ========= H = 100 model ===========
% (1) for S = 50, 
%       when sigma = 2, totEps= 1.2225

% (2) for S = 100,
%       when sigma = 2, totEps = 1.752

% (3) for S = 200,
%       when sigma = 2, totEps = 2.5267



%================================= sigma=1 ========================

% ========= H = 50 model ===========
% (1) for S = 50, 
%       when sigma = 1, totEps= 3.0697

% (2) for S = 100,
%       when sigma = 1, totEps = 4.4951

% (3) for S = 200,
%       when sigma = 1, totEps = 6.6063


% ========= H = 100 model ===========
% (1) for S = 50, 
%       when sigma = 1, totEps= 3.1612

% (2) for S = 100,
%       when sigma = 1, totEps = 4.6319

% (3) for S = 200,
%       when sigma = 1, totEps = 6.8171

%%


H50_TestAcc = zeros(length(Smat), length(Itermat), length(seednummat));
H100_TestAcc = zeros(length(Smat), length(Itermat), length(seednummat));


for seednum = seednummat
    for K=Hmat
        for maxit=1:length(Itermat)
            for S = Smat
                if S==20
                    if K==50
                        if sigma==2
                            %totEps = 1.1862;
%                             totEps = 0.38484;
                            totEps = 0.32944;
                            [S, K, sigma]
                        else % sigma = 1
                            %totEps = 3.0697;
                            % totEps = 1.5369;
                            totEps = 1.3166;
                        end
                    else % K==100
                        if sigma ==2
                            %totEps = 1.2225;
%                             totEps = 0.38484;
                            totEps = 0.32944;
                        else
                            %totEps = 3.1612;
%                             totEps = 1.5369;
                            totEps = 1.3166; 
                        end
                    end
                else % S==400
                    if K==50
                        if sigma ==2
                            %totEps = 1.7005;
%                             totEps = 0.42064;
                            totEps = 0.58159;
                            [S, K, sigma]
                        else
                            %totEps = 4.4951;
%                             totEps = 1.5388;
                            totEps = 2.3135;
                        end
                    else 
                        if sigma ==2
                            %totEps = 1.752;
%                             totEps = 0.42064;
                            totEps = 0.58159;
                        else
                            %totEps =  4.6319;
%                             totEps = 1.5388;
                            totEps = 2.3135;
                        end
                    end
                end
                filename = ['pri_mn_seed=' num2str(seednum) '_nH=' num2str(K) '_N=' num2str(Itermat(maxit)) '_S=' num2str(S) 'eps=' num2str(totEps) '_whichComp=' num2str(whichComp) '_sigma=' num2str(sigma) '.mat']; 
%                 display(filename)
                load(filename)
                if K==50
                    if S==20
                        H50_TestAcc(1, maxit, seednum) = result_sbnvb.TestAcc(end);
                    elseif S==50
                        H50_TestAcc(2, maxit, seednum) = result_sbnvb.TestAcc(end);
                    elseif S==100
                        H50_TestAcc(3, maxit, seednum) = result_sbnvb.TestAcc(end);
                    else
                        H50_TestAcc(4, maxit, seednum) = result_sbnvb.TestAcc(end);
                    end
                else % K==100
                    if S==20
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

H50_strong = squeeze(H50_TestAcc)';
H50_strong =  H50_strong(:,[1,4]);
figure(1);
subplot(1,3,1);
boxplot(H50_strong);
set(gca, 'ylim', [0.4, 1]);
 title('strong');
% set(gca, 'xtick', Smat);

H100_strong = squeeze(H100_TestAcc)';
H100_strong =  H100_strong(:,[1,4]);
figure(2);
subplot(1,3,1);
boxplot(H100_strong);
set(gca, 'ylim', [0.4, 1]); 
 title('strong');

figure(4);
W = result_sbnvb.W;
dispims(W,28,28); title('strong');

%% (3) private version : MA

% seednummat = linspace(1, 10, 10); 
seednumat = [1,2,3,4]; 
Hmat = [50, 100];
% Smat = [50, 100, 200];
Itermat = [0.25].*60000; % percent of the entire data

whichComp = 0; % MA

% only load files with sigma = 2

H50_TestAcc = zeros(length(Smat), length(Itermat), length(seednummat));
H100_TestAcc = zeros(length(Smat), length(Itermat), length(seednummat));


for seednum = seednummat
    for K=Hmat
        for maxit=1:length(Itermat)
            for S = Smat
                if S==20
                    if K==50
                        if sigma==2
                            %totEps = 1.1862;
%                             totEps = 0.38484;
                            totEps = 0.32944;
                            [S, K, sigma]
                        else % sigma = 1
                            %totEps = 3.0697;
                            % totEps = 1.5369;
                            totEps = 1.3166;
                        end
                    else % K==100
                        if sigma ==2
                            %totEps = 1.2225;
%                             totEps = 0.38484;
                            totEps = 0.32944;
                        else
                            %totEps = 3.1612;
%                             totEps = 1.5369;
                            totEps = 1.3166; 
                        end
                    end
                else % S==400
                    if K==50
                        if sigma ==2
                            %totEps = 1.7005;
%                             totEps = 0.42064;
                            totEps = 0.58159;
                            [S, K, sigma]
                        else
                            %totEps = 4.4951;
%                             totEps = 1.5388;
                            totEps = 2.3135;
                        end
                    else 
                        if sigma ==2
                            %totEps = 1.752;
%                             totEps = 0.42064;
                            totEps = 0.58159;
                        else
                            %totEps =  4.6319;
%                             totEps = 1.5388;
                            totEps = 2.3135;
                        end
                    end
                end
                filename = ['pri_mn_seed=' num2str(seednum) '_nH=' num2str(K) '_N=' num2str(Itermat(maxit)) '_S=' num2str(S) 'eps=' num2str(totEps) '_whichComp=' num2str(whichComp) '_sigma=' num2str(sigma) '.mat']; 
%                 display(filename)
                load(filename)
                if K==50
                    if S==20
                        H50_TestAcc(1, maxit, seednum) = result_sbnvb.TestAcc(end);
                    elseif S==50
                        H50_TestAcc(2, maxit, seednum) = result_sbnvb.TestAcc(end);
                    elseif S==100
                        H50_TestAcc(3, maxit, seednum) = result_sbnvb.TestAcc(end);
                    else
                        H50_TestAcc(4, maxit, seednum) = result_sbnvb.TestAcc(end);
                    end
                else % K==100
                    if S==20
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

H50_MA = squeeze(H50_TestAcc)';
H50_MA =  H50_MA(:,[1,4]);
figure(1);
subplot(1,3,2);
boxplot(H50_MA);
set(gca, 'ylim', [0.4, 1]);
 title('MA');
set(gca, 'xtick', Smat);

H100_MA = squeeze(H100_TestAcc)';
H100_MA =  H100_MA(:,[1,4]);
figure(2);
subplot(1,3,2);
boxplot(H100_MA);
set(gca, 'ylim', [0.4, 1]); 
 title('MA');

figure(6);
W = result_sbnvb.W;
dispims(W,28,28); title('MA');

