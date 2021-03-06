% labels
nlab = 2;
lbl_krapiv = [1 0];
lbl_smallw = [0 1];

N_total = N_krapiv + N_smallw;

if(exist('xval','var'))
assert(mod(N_total,xval) == 0, ...
       sprintf("N_total (currently %d) must be divisible by xval (currently %d)", ...
	       N_total,xval));
end
x_lab = cell(1,nlab);

%% krapivsky [1 0]
fprintf(1,'\nPopulating training data for model type %s.\n', 'krapivsky');
x_lab{logical(lbl_krapiv)} = load_krapivsky(z,
					    N_krapiv,
					    1000,
					    'data/krapivsky-networks/rdbn-%d.csv');
					    

%% smallworld [0 1]
fprintf(1,'\nPopulating training data for model type %s.\n', 'smallworld');
x_lab{logical(lbl_smallw)} = load_smallworld(z, N_smallw, 3, 0.2);

x = sparse([x_lab{logical(lbl_krapiv)}; ...
     x_lab{logical(lbl_smallw)}]);
clear x_lab;

labels = sparse([repmat(lbl_krapiv,N_krapiv,1); ...
	  repmat(lbl_smallw,N_smallw,1)]);

%% randomize order
ordr = randperm(N_total);
x = x(ordr,:);
labels = labels(ordr,:);
