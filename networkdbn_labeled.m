addpath(genpath('.'));
%addpath(genpath('/Users/jatwood/cnrg/graph-generation/deep-belief-gen'));
%addpath(genpath('data'));
%addpath(genpath('src'));

%addtosystempath('/opt/local/bin');

input_type='mix';

% compare samples to training?
compare = 1;

% training data parameters
N = 32;  % number of training networks (must be perfect square)
S = 1000; % number of nodes in the training networks (if pre-generated)
z = 200;   % only consider the z by z upper lefthand sub-matrix

% smallword parameters
smallworld_k = 3;
smallworld_p = 0.2;

% DBN parameters
L = 3;
K = [100 100 100];
T  = 50;
Tb = 50;
B = 8;
C = 100;
G = 10;
Gs = 5;
alpha = 0.1;
lambda = 0.0001;

sample_burn = 0;


% populate training data
fprintf(1,'\nPopulating training data for model type %s.\n', 'krapivsky');
x1 = zeros(N,z^2);
for i=1:N
  full_network = load_edgelist(sprintf('data/krapivsky-networks/rdbn-%d.csv',i),S);
  x1(i,:) = reshape(full_network(1:z,1:z),1,z^2);
end

fprintf(1,'\nPopulating training data for model type %s.\n', 'smallworld');
x2 = zeros(N,z^2);
for i=1:N
  full_network = full(smallw(z,smallworld_k,smallworld_p));
  x2(i,:) = reshape(full_network,1,z^2);
end

x = [x1; x2];
%labels = [zeros(N,1); ones(N,1)];
labels = zeros(2*N,2);
labels(1:N,1) = 1;
labels(N+1:end,2) = 1;
%labels
% [1 0]: krapivsky
% [0 1]: smallworld

ordr = randperm(2*N);
x = x(ordr,:);
labels = labels(ordr,:);

% train dbn
fprintf(1,'\nPretraining and backfitting dbn.\n');
dbn = dbntrain_labeled_real(x, labels, L, T, Tb, B, C, K, G, alpha, lambda);

if compare
   fprintf(1,'\nGenerating comparison plots.\n');
   timestamp = now;
    
   samples = dbnsample(dbn,2*N,Gs,sample_burn);
    
   fig=figure();
   for i=1:2*N
     subplot(sqrt(2*N),sqrt(2*N),i), imshow(reshape(x(i,:),z,z));
     title(sprintf('x(%d,:)',i));
   end
   saveas(fig,sprintf('results/dbn_%f_%s_x.pdf',timestamp,input_type),'pdf');
   
   fig=figure();
   for i=1:2*N
     subplot(sqrt(2*N),sqrt(2*N),i), imshow(reshape(samples(i,:),z,z));
     title(sprintf('s(%d,:)',i));
   end
   saveas(fig,sprintf('results/dbn_%f_%s_s.pdf',timestamp,input_type),'pdf');

   krapivsky_samples = dbnsample_clamp(dbn, [1,0], 2*N, Gs, sample_burn);
   
   fig=figure();
   for i=1:2*N
     subplot(sqrt(2*N),sqrt(2*N),i), imshow(reshape(krapivsky_samples(i,:),z,z));
     title(sprintf('krpv(%d,:)',i));
   end
   saveas(fig,sprintf('results/dbn_%f_%s_krapiv_s.pdf',timestamp,input_type),'pdf');

   sworld_samples = dbnsample_clamp(dbn, [0,1], 2*N, Gs, sample_burn);
   fig=figure();
   for i=1:2*N
     subplot(sqrt(2*N),sqrt(2*N),i), imshow(reshape(sworld_samples(i,:),z,z));
     title(sprintf('swrld(%d,:)',i));
   end
   saveas(fig,sprintf('results/dbn_%f_%s_sworld_s.pdf',timestamp,input_type),'pdf');

   for i=1:3
     plot_degree(x(i,:),samples(i,:),sprintf('results/dbn_%f_%s_degree_%d.pdf',timestamp,input_type,i));
   end

   fig = figure();
   genfield = dbn.gen.pair{1};
   for i=1:dbn.K(1)
       subplot(ceil(sqrt(K(1))),ceil(sqrt(K(1))),i), imshow(reshape(genfield(i,:),z,z));
       title(sprintf('gen',i));
   end
   saveas(fig,sprintf('results/dbn_%f_%s_receptive.pdf',timestamp,input_type),'pdf')
end
