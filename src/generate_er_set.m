N = 3000
z = 1000

for i=1:N
  full_network = full(erdrey(z,2*z));
  save(sprintf('data/er-networks/er%d',i),'full_network');
end
