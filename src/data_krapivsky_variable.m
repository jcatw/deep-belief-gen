N_total = N_krapiv;

if(exist('xval','var'))
assert(mod(N_total,xval) == 0, ...
       sprintf('N_total (currently %d) must be divisible by xval (currently %d)', ...
	       N_total,xval));
end

%% krapivsky 
fprintf(1,'\nPopulating training data for model type %s.\n', 'krapivsky');
x = sparse(load_krapivsky(z, ...
			  N_krapiv, ...
			  1000, ...
			  'data/krapivsky-networks-variable/set1/var-%d-edges.csv'));

fprintf(1,'\nPopulating training fitnesses for model type %s.\n', 'krapivsky');
fitness = load_fitness(z, ...
		       N_krapiv, ...
		       'data/krapivsky-networks-variable/set1/var-%d-nodes.csv');

%% randomize order
ordr = randperm(N_total);
x = x(ordr,:);
fitness = fitness(ordr,:);

