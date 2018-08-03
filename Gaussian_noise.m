function noise = Gaussian_noise(eps, del, len, sen)

c2 = 2*log(1.25/del);
nsv = c2*sen^2/(eps^2);
if length(len)==1
    noise = normrnd(0, sqrt(nsv), len, 1);
else
    noise = normrnd(0, sqrt(nsv), len(1), len(2));
end