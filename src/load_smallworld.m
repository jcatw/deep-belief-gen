function [smallw_x] = load_smallworld(z, N_smallw, smallworld_k, smallworld_p)

smallw_x = zeros(N_smallw,z^2);
for i=1:N_smallw
  full_network = smallw(z,smallworld_k,smallworld_p);
  %full_network = load(sprintf('data/smallworld-networks/sw%d',i));
  smallw_x(i,:) = reshape(full_network(1:z,1:z),1,z^2);
  %smallw_x(i,:) = reshape(full_network.full_network(1:z,1:z),1,z^2);
end
