function [iterEps, iterDel]= cal_amp_eps(numIter, whichComp, totEps, totDel, sampRate)

% numIter
% whichComp: 0 (linear), 1 (adv), 2 (zCDP)
% totEps
% totDel
% sampRate: minibatchsize/totalDatasize

%% 
x0 = 0.1;

if whichComp==0 % linear composition 
   
    fun = @(x) (totEps - numIter*log(1 + sampRate*(exp(x)-1)))^2; 
    iterEps = fmincon(fun,x0,[], [] ,[],[],1e-6,[]);
    iterDel = totDel/numIter;
%     eps_amp = np.log(1 + nu*(np.exp(x)-1))
elseif whichComp == 1 % Advanced composition 
    iterDel = 0.0001;
    delta = 0.0001; 
%     - numIter*sampRate*iterDel;
    if delta<0
        display('delta is less than 0')
    end
%     eps_amp = log(1 + sampRate*(exp(x)-1))
    fun = @(x) (totEps - sqrt(2*numIter*log(1/delta))*log(1 + sampRate*(exp(x)-1)) - numIter*log(1 + sampRate*(exp(x)-1))*(exp(log(1 + sampRate*(exp(x)-1)))-1))^2;
    iterEps = fmincon(fun,x0,[], [] ,[],[],1e-6,[]);

else % whichComp==2,  zCDP composition 
    iterDel = 0.0001;
    delta_amp = sampRate*iterDel;
    c2 = 2*log(1.25/delta_amp);
    delta = 0.0001;
%     rho = numIter*(eps_amp^2)/(2*c2)

    fun = @(x)(totEps - (numIter*((log(1 + sampRate*(exp(x)-1)))^2)/(2*c2) + 2*sqrt(numIter*((log(1 + sampRate*(exp(x)-1)))^2)/(2*c2)*log(1/delta))))^2; 
    iterEps = fmincon(fun,x0,[], [] ,[],[],1e-6,[]);
end
