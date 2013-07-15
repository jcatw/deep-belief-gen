function [krapiv_x] = load_krapivsky(z,N_krapiv,krapiv_S,fmt_string)

krapiv_x = zeros(N_krapiv,z^2);
for i=1:N_krapiv
  full_network = load_edgelist(sprintf(fmt_string,i),krapiv_S);
  krapiv_x(i,:) = reshape(full_network(1:z,1:z),1,z^2);
end
			  
