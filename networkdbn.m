addpath(genpath('/Users/jatwood/cnrg/graph-generation/deep-belief-gen'));
%addpath(genpath('data'));
%addpath(genpath('src'));

addtosystempath('/opt/local/bin');

% type of input network
% one of:
%  'krapivsky'
%  'smallworld'
input_type='krapivsky'

% compare samples to training?
compare = 0;

% training data parameters
N = 49;  % number of training networks (must be perfect square)
S = 1000; % number of nodes in the training networks (if pre-generated)
z = 200;   % only consider the z by z upper lefthand sub-matrix

% smallword parameters
smallworld_k = 3;
smallworld_p = 0.2;

% DBN parameters
L = 3;
K = [100 50 50];
T = 50;
B = 7;
C = 100;
G = 4;
alpha = 0.1;
lambda = 0.0001;


% populate training data
if strcmp(input_type,'krapivsky')
    x = zeros(N,z^2);
    for i=1:N
        full_network = load_edgelist(sprintf('data/krapivsky-networks/rdbn-%d.csv',i),S);
        x(i,:) = reshape(full_network(1:z,1:z),1,z^2);
    end
elseif strcmp(input_type,'smallworld')
    x = zeros(N,z^2);
    for i=1:N
        full_network = full(smallw(z,smallworld_k,smallworld_p));
        x(i,:) = reshape(full_network,1,z^2);
    end
end

% train dbn
dbn = dbntrain(x, L, T, B, C, K, G, alpha, lambda);

if compare
    timestamp = now;
    
    samples = dbnsample(dbn,N,G);
    
    fig=figure();
    for i=1:N
        subplot(sqrt(N),sqrt(N),i), imshow(reshape(x(N,:),z,z));
        title(sprintf('x(%d,:)',i));
    end
    saveas(fig,sprintf('results/dbn_%f_%s_x.pdf',timestamp,input_type),'pdf');
    
    fig=figure();
    for i=1:N
        subplot(sqrt(N),sqrt(N),i), imshow(reshape(samples(N,:),z,z));
        title(sprintf('s(%d,:)',i));
    end
    saveas(fig,sprintf('results/dbn_%f_%s_s.pdf',timestamp,input_type),'pdf');
    
    
end
