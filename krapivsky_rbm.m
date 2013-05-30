addpath(genpath('/Users/jatwood/cnrg/graph-generation/deep-belief-gen'));
%addpath(genpath('data'));
%addpath(genpath('src'));

addtosystempath('/opt/local/bin');

% plot?
plot = 0;

% training data parameters
N = 100;  % number of training networks (must be perfect square)
S = 1000; % number of nodes in the training networks
z = 100;   % only consider the z by z upper lefthand sub-matrix

% RBM parameters
K = 1000;
T = 50;
B = 10;
C = 100;
alpha = 0.1;
lambda = 0.0001;


% populate training data
x = zeros(N,z^2);
for i=1:N
    full_network = load_edgelist(sprintf('data/krapivsky-networks/rdbn-%d.csv',i),S);
    x(i,:) = reshape(full_network(1:z,1:z),1,z^2);
end

% train rbm
rbm = rbmtrain(x, T, B, C, K, alpha, lambda);

% visualize training data, receptive fields
if plot
    figure()
    for i=1:N
        subplot(sqrt(N),sqrt(N),i), imshow(reshape(x(N,:),z,z));
        title(sprintf('x(%d,:)',i));
    end
    %title('Krapivsky Training Adjacency Matrices');
    %text(.75,1.25,'Krapivsky Training Adjacency Matrices')

    Wp_t = rbm.Wp';
    figure()
    for i=1:K
        subplot(sqrt(K),sqrt(K),i), imshow(reshape(Wp_t(i,:),z,z));
        title(sprintf('Wp`(%d,:)',i));
    end
    %title('Krapivsky RBM Receptive Field');
    %text(0.75,1.25,'Krapivsky RBM Receptive Field');
end