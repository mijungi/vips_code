function noise = AnalyzeGauss(eps, del, len, sen)

c2 = 2*log(1.25/del);
nsv = c2*sen^2/(eps^2);

noise = normrnd(0, sqrt(nsv), len, len);

upp = triu(noise, 1);
lower = upp'; 
noise = triu(noise,0) + lower; 