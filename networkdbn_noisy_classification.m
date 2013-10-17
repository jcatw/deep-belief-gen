addpath(genpath('.'));

timestamp=now;
writefilename = 'classification_results_noisy.csv';
writefile = fopen(writefilename,'a+');

xval=5;

% DBN, simulation parameters
%standard_small_parameterization;
%random_parameterization;
%random_parameterization_big;
%super_small;
small_successful_parameterization;
%big_successful_parameterization;

total_accuracy = zeros(1,10);
iter = 1;
for noise_level=0.0:0.1:1.0
    clear x;
    clear ordr;
    clear labels;

    data_krapivsky_er_noisy;
    
    accuracy = zeros(1,xval);
    N_xval = N_total / xval;

    for i=1:xval
      x_train = [x(1:(i-1) * N_xval,:); ...
				      x(1 + i*N_xval : end,:)];
      x_test = x( 1+(i-1)*N_xval : i*N_xval,:);
      labels_train = [labels(1:(i-1) * N_xval,:); ...
						labels(1 + i*N_xval : end,:)];
      labels_test = labels( 1+(i-1)*N_xval : i*N_xval,:);

      %size(x_train)
      %size(labels_train)

      fprintf(1,'\nValidation %d: Pretraining and backfitting dbn (noise level %f).\n',i,noise_level);
      clear dbn;
      dbn = dbntrain(x_train, L, T, Tb, B, C, K, G, alpha, lambda, labels_train);
      %save(sprintf('results/dbn_%f_%d.mat',timestamp,i),'dbn');
      predictions = dbnclassify(dbn,x_test,10);

      accuracy(i) = sum(all(predictions == labels_test,2)) / N_xval;
      fprintf(1,'\nValidation %d Accuracy: %f (noise level %f)\n\n', ...
	      i, ...
	      100*accuracy(i), ...
	      noise_level);
    end
    total_accuracy(iter) = mean(accuracy);
    fprintf(1, '\n%f noise level %d-fold cross-validation accuracy: %f\n\n', noise_level, xval, 100*total_accuracy(iter));
    fprintf(writefile, '%f,%f\n', noise_level, total_accuracy(iter));
    iter = iter+1;
    
end

