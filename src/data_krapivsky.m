N_total = N_krapiv;

if(exist('xval','var'))
assert(mod(N_total,xval) == 0, ...
       sprintf('N_total (currently %d) must be divisible by xval (currently %d)', ...
	       N_total,xval));
end

%% krapivsky 
%fprintf(1,'\nPopulating training data for model type %s.\n', 'krapivsky');
%x = sparse(load_krapivsky(z, ...
%			  N_krapiv, ...
%			  1000, ...
%			  'data/krapivsky-networks/rdbn-%d.csv'));

%fprintf(1,'\nPopulating training data for model type %s.\n', 'krapivsky-same');
%x = sparse(load_krapivsky(z, ...
%			  N_krapiv, ...
%			  1000, ...
%			  'data/krapivsky-networks-sitevisit/same/same-%d.csv'));
%resfilebase = 'sitevisit/mat/same/%s'

fprintf(1,'\nPopulating training data for model type %s.\n', 'krapivsky-different');
x = sparse(load_krapivsky(z, ...
			  N_krapiv, ...
			  1000, ...
			  'data/krapivsky-networks-sitevisit/different/different-%d.csv'));
resfilebase = 'sitevisit/mat/different/%s'

%% randomize order
ordr = randperm(N_total);
x = x(ordr,:);
