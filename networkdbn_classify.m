
addpath(genpath('.'));

timestamp=now;
writefilename = 'classification_results_many_parameterizations.csv';

many_parameterizations = 1;

% xval-fold cross-validation
xval = 5;

% DBN, simulation parameters
%standard_small_parameterization;
%random_parameterization;
%random_parameterization_big;
%super_small;
small_successful_parameterization;
%big_successful_parameterization;

% Dataset
if(many_parameterizations)
  data_krapivsky_smallworld_er_many;
else
  data_krapivsky_smallworld_er;
end

% cross-validate
accuracy = zeros(1,xval);
N_xval = N_total / xval;

all_predictions = zeros(N_total, nlab);
if(many_parameterizations)
  rough_accuracy = zeros(1,xval);
  all_rough_predictions = zeros(N_total,nlab/3);
end

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
    dbn = dbntrain(x_train, L, T, Tb, B, C, K, G, alpha, lambda, labels_train);
    save(sprintf('results/dbn_%f_%d.mat',timestamp,i),'dbn');
    predictions = dbnclassify(dbn,x_test,10);
    %size(predictions)
    %size(labels_test)
    %size(predictions == labels_test)
    %size(sum(predictions == labels_test))
    accuracy(i) = sum(all(predictions == labels_test,2)) / N_xval;
    fprintf(1,'\nValidation %d Accuracy: %f\n\n', i, 100*accuracy(i));

    all_predictions(1+(i-1)*N_xval : i*N_xval,:) = predictions;
    
    if(many_parameterizations)
      rough_predictions = [sum(predictions(:,1:3),2),
			   sum(predictions(:,4:6),2),
			   sum(predictions(:,7:9),2)];
      rough_labels = [sum(labels_test(:,1:3),2),
		      sum(labels_test(:,4:6),2),
		      sum(labels_test(:,7:9),2)];
      rough_accuracy(i) = sum(all(rough_predictions == rough_labels,2)) / N_xval;
      all_rough_predictions(1+(i-1)*N_xval : i*N_xval,:) = rough_predictions;
    end
      
      
end
fprintf(1, '\n%d-fold cross-validation accuracy: %f\n\n', xval, 100*mean(accuracy));

confusion_matrix = confusion(labels, all_predictions)

%fig=figure();
%imshow(confusion_matrix);
%title('Confusion Matrix');
%saveas(fig,sprintf('results/dbn_%f_classification_confusion.pdf',timestamp),'pdf');

writefile = fopen(writefilename,'a+');
fprintf(writefile, ...
	'%d,%d,%d,%d,%d,%d,%d,%f,%f,%f\n', ...
	N_total,z,L,K(1),T,B,G,alpha,lambda,mean(accuracy));
fclose(writefile);
