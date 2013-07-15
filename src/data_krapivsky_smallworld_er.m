% labels
nlab = 3;
lbl_krapiv = [1 0 0];
lbl_smallw = [0 1 0];
lbl_er     = [0 0 1];

% training data
N = zeros(1,nlab);  % number of training instances by label
%z = 200;            % number of nodes in each network

N(logical(lbl_krapiv)) = N_krapiv;
N(logical(lbl_smallw)) = N_smallw;
N(logical(lbl_er))     = N_er;
N_total = N(logical(lbl_krapiv)) + N(logical(lbl_smallw)) + N(logical(lbl_er));

if(exist('xval','var'))
assert(mod(N_total,xval) == 0, ...
       sprintf("N_total (currently %d) must be divisible by xval (currently %d)", ...
	       N_total,xval));
end
x_lab = cell(1,nlab);

%% krapivsky [1 0 0]
krapiv_S = 1000;
fprintf(1,'\nPopulating training data for model type %s.\n', 'krapivsky');

krapiv_x = zeros(N(logical(lbl_krapiv)),z^2);
for i=1:N(logical(lbl_krapiv))
  full_network = load_edgelist(sprintf('data/krapivsky-networks/rdbn-%d.csv',i),krapiv_S);
  krapiv_x(i,:) = reshape(full_network(1:z,1:z),1,z^2);
end
x_lab{logical(lbl_krapiv)} = krapiv_x;
clear krapiv_x;

%% smallworld [0 1 0]
smallworld_k = 3;
smallworld_p = 0.2;

fprintf(1,'\nPopulating training data for model type %s.\n', 'smallworld');

smallw_x = zeros(N(logical(lbl_smallw)),z^2);
for i=1:N(logical(lbl_smallw))
  full_network = smallw(z,smallworld_k,smallworld_p);
  %full_network = load(sprintf('data/smallworld-networks/sw%d',i));
  smallw_x(i,:) = reshape(full_network(1:z,1:z),1,z^2);
  %smallw_x(i,:) = reshape(full_network.full_network(1:z,1:z),1,z^2);
  
end
x_lab{logical(lbl_smallw)} = smallw_x;
clear smallw_x;
clear full_network;

%% er [0 0 1]
fprintf(1,'\nPopulating training data for model type %s.\n', 'erdos-renyi');
%fprintf(1,'\nPopulating training data for model type %s.\n', 'sticky');

er_x = zeros(N(logical(lbl_er)),z^2);
for i=1:N(logical(lbl_er))
  full_network = erdrey(z,2*z);
  %full_network = load(sprintf('data/er-networks/er%d',i));
  er_x(i,:) = reshape(full_network(1:z,1:z),1,z^2);
  %er_x(i,:) = reshape(full_network.full_network(1:z,1:z),1,z^2);
end
x_lab{logical(lbl_er)} = er_x;
clear er_x;
clear full_network;

x = sparse([x_lab{logical(lbl_krapiv)}; ...
     x_lab{logical(lbl_smallw)}; ...
     x_lab{logical(lbl_er)}      ]);
clear x_lab;

labels = sparse([repmat(lbl_krapiv,N(logical(lbl_krapiv)),1); ...
	  repmat(lbl_smallw,N(logical(lbl_smallw)),1); ...
	  repmat(lbl_er,N(logical(lbl_er)),1)          ]);

%% randomize order
ordr = randperm(N_total);
x = x(ordr,:);
labels = labels(ordr,:);
