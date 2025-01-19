function [ nestPop ] = func_newBuildNest( nestPop ,Pa ,ParticleScope)
Xmin=ParticleScope(1,:);
Xmax=ParticleScope(2,:);
[N,D] = size(nestPop) ;
Xmax=repmat(Xmax,N,1);
Xmin=repmat(Xmin,N,1);
nestPop = nestPop+rand.*heaviside(rand(N,D)-Pa).*(nestPop(randperm(N),:)-nestPop(randperm(N),:));
nestPop((nestPop>Xmax)) = Xmax((nestPop>Xmax)) ;
nestPop((nestPop<Xmin)) = Xmin((nestPop<Xmin)) ;
end

