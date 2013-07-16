% labels
nlab = 3;
lbl_krapiv_1 = [1 0 0 0 0 0 0 0 0];
lbl_krapiv_2 = [0 1 0 0 0 0 0 0 0];
lbl_krapiv_3 = [0 0 1 0 0 0 0 0 0];

lbl_smallw_1 = [0 0 0 1 0 0 0 0 0];
lbl_smallw_2 = [0 0 0 0 1 0 0 0 0];
lbl_smallw_3 = [0 0 0 0 0 1 0 0 0];

lbl_er_1 = [0 0 0 0 0 0 1 0 0];
lbl_er_2 = [0 0 0 0 0 0 0 1 0];
lbl_er_3 = [0 0 0 0 0 0 0 0 1];

N_total = N_krapiv + N_smallw + N_er;

if(exist('xval','var'))
assert(mod(N_total,xval) == 0, ...
       sprintf("N_total (currently %d) must be divisible by xval (currently %d)", ...
	       N_total,xval));
end
x_lab = cell(1,nlab);

%% krapivsky [1 1 1 0 0 0 0 0 0]
fprintf(1,'\nPopulating training data for model type %s.\n', 'krapivsky');
x_lab{logical(lbl_krapiv_1)} = load_krapivsky(z,
					      N_krapiv / 3,
					      1000,
					      'data/krapivsky-networks-param/set1/rdbn-%d.csv');

x_lab{logical(lbl_krapiv_2)} = load_krapivsky(z,
					      N_krapiv / 3,
					      1000,
					      'data/krapivsky-networks-param/set2/rdbn-%d.csv');

x_lab{logical(lbl_krapiv_3)} = load_krapivsky(z,
					      N_krapiv / 3,
					      1000,
					      'data/krapivsky-networks-param/set3/rdbn-%d.csv');
					    

%% smallworld [0 0 0 1 1 1 0 0 0]
fprintf(1,'\nPopulating training data for model type %s.\n', 'smallworld');
x_lab{logical(lbl_smallw_1)} = load_smallworld(z,
					     N_smallw / 3,
					     3,
					     0.2);

x_lab{logical(lbl_smallw_2)} = load_smallworld(z,
					     N_smallw / 3,
					     3,
					     0.1);

x_lab{logical(lbl_smallw_3)} = load_smallworld(z,
					     N_smallw / 3,
					     3,
					     0.4);


%% er [0 0 0 0 0 0 1 1 1]
fprintf(1,'\nPopulating training data for model type %s.\n', 'erdos-renyi');
%fprintf(1,'\nPopulating training data for model type %s.\n', 'sticky');
x_lab{logical(lbl_er_1)} = load_er(z,
				   N_er,
				   z,
				   z^2);

x_lab{logical(lbl_er_2)} = load_er(z,
				   N_er,
				   z,
				   z^3);

x_lab{logical(lbl_er_3)} = load_er(z,
				   N_er,
				   z,
				   z);

x = sparse([x_lab{logical(lbl_krapiv_1)}; ...
     x_lab{logical(lbl_krapiv_2)}; ...
     x_lab{logical(lbl_krapiv_3)}; ...
     x_lab{logical(lbl_smallw_1)}; ...
     x_lab{logical(lbl_smallw_2)}; ...
     x_lab{logical(lbl_smallw_3)}; ...
     x_lab{logical(lbl_er_1)}; ...
     x_lab{logical(lbl_er_2)}; ...
     x_lab{logical(lbl_er_3)}      ]);
clear x_lab;

labels = sparse([repmat(lbl_krapiv_1,N_krapiv,1); ...
	  repmat(lbl_krapiv_2,N_krapiv,1); ...
	  repmat(lbl_krapiv_3,N_krapiv,1); ...
	  repmat(lbl_smallw_1,N_smallw,1); ...
	  repmat(lbl_smallw_2,N_smallw,1); ...
	  repmat(lbl_smallw_3,N_smallw,1); ...
	  repmat(lbl_er_1,N_er,1); ...
	  repmat(lbl_er_2,N_er,1); ...
	  repmat(lbl_er_3,N_er,1)          ]);

%% randomize order
ordr = randperm(N_total);
x = x(ordr,:);
labels = labels(ordr,:);
