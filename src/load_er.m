function [er_x] = load_er(z, N_er, m, n)

er_x = zeros(N_er,z^2);
for i=1:N_er
  full_network = erdrey(z,2*z);
  %full_network = load(sprintf('data/er-networks/er%d',i));
  er_x(i,:) = reshape(full_network(1:z,1:z),1,z^2);
  %er_x(i,:) = reshape(full_network.full_network(1:z,1:z),1,z^2);
end
