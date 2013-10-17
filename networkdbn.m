addpath(genpath('.'));
%addpath(genpath('/Users/jatwood/cnrg/graph-generation/deep-belief-gen'));
%addpath(genpath('data'));
%addpath(genpath('src'));

%addtosystempath('/opt/local/bin');

% type of input network
% one of:
%  'krapivsky'
%  'smallworld'
input_type='krapivsky';

% compare samples to training?
compare = 1;

%small_successful_parameterization;
smallest_successful_model;
%super_small;

%% training data parameters
%N = 49;  % number of training networks (must be perfect square)
%S = 1000; % number of nodes in the training networks (if pre-generated)
%z = 200;   % only consider the z by z upper lefthand sub-matrix
%
%% smallword parameters
%smallworld_k = 3;
%smallworld_p = 0.2;
%
%% DBN parameters
%L = 3;
%K = [100 50 50];
%T = 50;
%B = 7;
%C = 100;
%G = 4;
%alpha = 0.1;
%lambda = 0.0001;




% populate training data
%fprintf(1,'\nPopulating training data for model type %s.\n', input_type);
%if strcmp(input_type,'krapivsky')
%    x = zeros(N,z^2);
%    for i=1:N
%        full_network = load_edgelist(sprintf('data/krapivsky-networks/rdbn-%d.csv',i),S);
%        x(i,:) = reshape(full_network(1:z,1:z),1,z^2);
%    end
%elseif strcmp(input_type,'smallworld')
%    x = zeros(N,z^2);
%    for i=1:N
%        full_network = full(smallw(z,smallworld_k,smallworld_p));
%        x(i,:) = reshape(full_network,1,z^2);
%    end
%end

data_krapivsky;

N = N_total;

% train dbn
fprintf(1,'\nPretraining and backfitting dbn.\n');
dbn = dbntrain(x, L, T, T, B, C, K, G, alpha, lambda);

if compare
   fprintf(1,'\nGenerating comparison plots.\n');
   timestamp = now;
    
   samples = dbnsample(dbn,N,G,10);
    
   %fig=figure();
   %for i=1:N
   %  subplot(sqrt(N),sqrt(N),i), imshow(reshape(x(i,:),z,z));
   %  title(sprintf('x(%d,:)',i));
   %end
   %saveas(fig,sprintf('results/dbn_%f_%s_x.pdf',timestamp,input_type),'pdf');
   %
   %fig=figure();
   %for i=1:N
   %  subplot(sqrt(N),sqrt(N),i), imshow(reshape(samples(i,:),z,z));
   %  title(sprintf('s(%d,:)',i));
   %end
   %saveas(fig,sprintf('results/dbn_%f_%s_s.pdf',timestamp,input_type),'pdf');

   %if strcmp(input_type,'krapivsky')
   %   for i=1:3
   %	plot_degree(x(i,:),samples(i,:),sprintf('results/dbn_%f_%s_degree_%d.pdf',timestamp,input_type,i));
   %   end
   %end

   data_beta_in = zeros(N,2);
   data_beta_out = zeros(N,2);
   
   sample_beta_in = zeros(N,2);
   sample_beta_out = zeros(N,2);
   for i=1:N
       [a_data_beta_in, a_data_beta_out] = fit_degree(x(i,:));
       [a_sample_beta_in, a_sample_beta_out] = fit_degree(samples(i,:));

       data_beta_in(i,:)  = a_data_beta_in;
       data_beta_out(i,:) = a_data_beta_out;
       sample_beta_in(i,:)  = a_sample_beta_in;
       sample_beta_out(i,:) = a_sample_beta_out;
   end

   data_alpha_in    = abs(data_beta_in(:,1)   ) + 1.0;
   data_alpha_out   = abs(data_beta_out(:,1)  ) + 1.0;
   sample_alpha_in  = abs(sample_beta_in(:,1) ) + 1.0;
   sample_alpha_out = abs(sample_beta_out(:,1)) + 1.0;

   fprintf(1,"\n");
   fprintf(1,"%s alpha %s: %f (%f)\n","data","in",   mean(data_alpha_in),std(data_alpha_in));
   fprintf(1,"%s alpha %s: %f (%f)\n","sample","in", mean(sample_alpha_in),std(sample_alpha_in));
   fprintf(1,"\n");
   fprintf(1,"%s alpha %s: %f (%f)\n","data","out",  mean(data_alpha_out),std(data_alpha_out));
   fprintf(1,"%s alpha %s: %f (%f)\n","sample","out",mean(sample_alpha_out),std(sample_alpha_out));
   fprintf(1,"\n");

   save(sprintf(resfilebase,'data_alpha_in'),'data_alpha_in');
   save(sprintf(resfilebase,'data_alpha_out'),'data_alpha_out');
   save(sprintf(resfilebase,'sample_alpha_in'),'sample_alpha_in');
   save(sprintf(resfilebase,'sample_alpha_out'),'sample_alpha_out');
end
