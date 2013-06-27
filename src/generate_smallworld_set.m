N = 3000
z = 1000
smallworld_k = 3
smallworld_p = 0.2

for i=1:N
    full_network = full(smallw(z,smallworld_k,smallworld_p));
    save(sprintf('data/smallworld-networks/sw%d',i),'full_network');
end
    
    
