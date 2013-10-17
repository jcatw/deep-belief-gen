
addpath(genpath('.'));

timestamp=now;
writefilename = 'classification_results_fitness.csv';

% DBN, simulation parameters
%standard_small_parameterization;
%random_parameterization;
%random_parameterization_big;
%super_small;
small_successful_parameterization;
%big_successful_parameterization;

% Dataset
data_krapivsky_fitness;

x_train = x[1:(3*N_total)/4 - 1];
fitness_train = fitness[1:(3*N_total)/4 - 1];
x_test = x[(3*N_total)/4:end];
fitness_test = fitness[(3*N_total)/4:end];

writefile = fopen(writefilename,'a+');
fprintf(writefile, ...
	'%d,%d,%d,%d,%d,%d,%d,%f,%f,%f\n', ...
	N_total,z,L,K(1),T,B,G,alpha,lambda,mean(accuracy));
fclose(writefile);
