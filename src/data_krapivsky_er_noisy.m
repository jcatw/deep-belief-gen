% labels
nlab = 2;
lbl_krapiv = [1 0];
lbl_er     = [0 1];

N_total = N_krapiv + N_er;

if(exist('xval','var'))
assert(mod(N_total,xval) == 0, ...
       sprintf("N_total (currently %d) must be divisible by xval (currently %d)", ...
	       N_total,xval));
end
x_lab = cell(1,nlab);

%% noisy krapivsky [1 0]
fprintf(1,'\nPopulating training data for model type %s.\n', 'krapivsky');
krapiv_pure = load_krapivsky(z,
			     (1 - noise_level) * N_krapiv,
			     1000,
			     'data/krapivsky-networks/rdbn-%d.csv');
krapiv_noise = load_er(z,noise_level * N_krapiv,z,z^2);
x_lab{logical(lbl_krapiv)} = [krapiv_pure; krapiv_noise];
clear krapiv_pure;
clear krapiv_noise;

%% er [0 1]
fprintf(1,'\nPopulating training data for model type %s.\n', 'erdos-renyi');

x_lab{logical(lbl_er)} = load_er(z,N_er,z,z^2);
x = sparse([x_lab{logical(lbl_krapiv)}; ...
     x_lab{logical(lbl_er)}      ]);
clear x_lab;

labels = sparse([repmat(lbl_krapiv,N_krapiv,1); ...
	  repmat(lbl_er,N_er,1)]);

%% randomize order
ordr = randperm(N_total);
x = x(ordr,:);
labels = labels(ordr,:);
