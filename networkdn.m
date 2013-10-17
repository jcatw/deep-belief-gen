addpath(genpath('.'));

N_krapiv = 300;
z = 200;
alpha = 0.05;

data_krapivsky_variable;

fprintf(1,'Fitting dependency network\n');
dn = dependency_network_fit(full(x),alpha);


