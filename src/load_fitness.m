function [fitness] = load_fitness(z,N_krapiv,fmt_string)

fitness = zeros(N_krapiv,z,2);
for i=1:N_krapiv
    nodelist = load(sprintf(fmt_string,i));
    fitness(i,:,:) = nodelist(1:z,2:3);
end


    
