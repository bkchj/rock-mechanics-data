function [ result ] = func_levy( nestPop,ParticleScope)
Xmin=ParticleScope(1,:);
Xmax=ParticleScope(2,:);
[N,D] = size(nestPop) ;
Xmax=repmat(Xmax,N,1);
Xmin=repmat(Xmin,N,1);
beta = 1.5 ;
alpha = 1 ;
sigma_u = (gamma(1+beta)*sin(pi*beta/2)/(beta*gamma((1+beta)/2)*2^((beta-1)/2)))^(1/beta) ;
sigma_v = 1 ;
u = normrnd(0,sigma_u,N,D) ;
v = normrnd(0,sigma_v,N,D) ;
step = u./(abs(v).^(1/beta)) ;
nestPop = nestPop+alpha.*step ;
nestPop((nestPop>Xmax)) = Xmax((nestPop>Xmax)) ;
nestPop((nestPop<Xmin)) = Xmin((nestPop<Xmin)) ;
result = nestPop ; 
end