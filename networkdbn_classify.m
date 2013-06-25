addpath(genpath('.'));

timestamp=now;

% xval-fold cross-validation
xval = 3;

% DBN parameters
L = 3;
K = [100 100 100];
T  = 20;
Tb = 20;
B = 5;
C = 100;
G = 10;
Gs = 5;
alpha = 0.1;
lambda = 0.0001;

% labels
nlab = 3;
lbl_krapiv = [1 0 0];
lbl_smallw = [0 1 0];
lbl_er     = [0 0 1];

% training data
N = zeros(1,nlab);  % number of training instances by label
z = 200;            % number of nodes in each network

N(logical(lbl_krapiv)) = 100;
N(logical(lbl_smallw)) = 100;
N(logical(lbl_er))     = 100;
N_total = N(logical(lbl_krapiv)) + N(logical(lbl_smallw)) + N(logical(lbl_er));

assert(mod(N_total,xval) == 0, ...
       sprintf("N_total (currently %d) must be divisible by xval (currently %d)", ...
	       N_total,xval));

x_lab = cell(1,nlab);

%% krapivsky [0 1 0]
krapiv_S = 1000;
fprintf(1,'\nPopulating training data for model type %s.\n', 'krapivsky');

krapiv_x = zeros(N(logical(lbl_krapiv)),z^2);
for i=1:N(logical(lbl_krapiv))
  full_network = load_edgelist(sprintf('data/krapivsky-networks/rdbn-%d.csv',i),krapiv_S);
  krapiv_x(i,:) = reshape(full_network(1:z,1:z),1,z^2);
end
x_lab{logical(lbl_krapiv)} = krapiv_x;
clear krapiv_x;

%% smallworld [1 0 0]
smallworld_k = 3;
smallworld_p = 0.2;

fprintf(1,'\nPopulating training data for model type %s.\n', 'smallworld');

smallw_x = zeros(N(logical(lbl_smallw)),z^2);
for i=1:N(logical(lbl_smallw))
  full_network = full(smallw(z,smallworld_k,smallworld_p));
  smallw_x(i,:) = reshape(full_network,1,z^2);
end
x_lab{logical(lbl_smallw)} = smallw_x;
clear smallw_x;

%% er [0 0 1]
fprintf(1,'\nPopulating training data for model type %s.\n', 'erdos-renyi');
%fprintf(1,'\nPopulating training data for model type %s.\n', 'sticky');

er_x = zeros(N(logical(lbl_er)),z^2);
for i=1:N(logical(lbl_er))
  full_network = full(erdrey(z,2*z));
  er_x(i,:) = reshape(full_network,1,z^2);
end
x_lab{logical(lbl_er)} = er_x;
clear er_x;

x = [x_lab{logical(lbl_krapiv)}; ...
     x_lab{logical(lbl_smallw)}; ...
     x_lab{logical(lbl_er)}      ];

labels = [repmat(lbl_krapiv,N(logical(lbl_krapiv)),1); ...
	  repmat(lbl_smallw,N(logical(lbl_smallw)),1); ...
	  repmat(lbl_er,N(logical(lbl_er)),1)          ];

%%size(x)
%%size(labels)

%% randomize order
ordr = randperm(N_total);
x = x(ordr,:);
labels = labels(ordr,:);

% cross-validate
accuracy = zeros(1,xval);
N_xval = N_total / xval;

all_predictions = zeros(N_total, nlab);

for i=1:xval
    x_train = [x(1:(i-1) * N_xval,:); ...
	       x(1 + i*N_xval : end,:)];
    x_test = x( 1+(i-1)*N_xval : i*N_xval,:);
    labels_train = [labels(1:(i-1) * N_xval,:); ...
		    labels(1 + i*N_xval : end,:)];
    labels_test = labels( 1+(i-1)*N_xval : i*N_xval,:);

    %size(x_train)
    %size(labels_train)

    fprintf(1,'\nValidation %d: Pretraining and backfitting dbn.\n',i);
    clear dbn;
    dbn = dbntrain_labeled(x_train, labels_train, L, T, Tb, B, C, K, G, alpha, lambda);
    predictions = dbnclassify(dbn,x_test,10);
    %size(predictions)
    %size(labels_test)
    %size(predictions == labels_test)
    %size(sum(predictions == labels_test))
    accuracy(i) = sum(all(predictions == labels_test,2)) / N_xval;
    fprintf(1,'\nValidation %d Accuracy: %f\n\n', i, 100*accuracy(i));

    all_predictions(1+(i-1)*N_xval : i*N_xval,:) = predictions;
end
fprintf(1, '\n%d-fold cross-validation accuracy: %f\n\n', xval, 100*mean(accuracy));

confusion_matrix = confusion(labels, all_predictions)

fig=figure();
imshow(confusion_matrix);
title('Confusion Matrix');
saveas(fig,sprintf('results/dbn_%f_classification_confusion.pdf',timestamp),'pdf');
