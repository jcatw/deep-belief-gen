space_L = [3 4 5 6 7];
space_K = [100 200 400 800 1600];
space_T = [100];
space_B = [10 20];
space_G = [5 10 20];
space_alpha = [0.1 0.01 0.001];
space_lambda = [0.1 0.01 0.001 0.0001];
space_N = [750 1500 3000];
space_z = [500 1000];

L = space_L(randi(length(space_L)))
K = repmat(space_K(randi(length(space_K))),1,L)
T = space_T(randi(length(space_T)))
B = space_B(randi(length(space_B)))
G = space_G(randi(length(space_G)))
alpha = space_alpha(randi(length(space_alpha)))
lambda = space_lambda(randi(length(space_lambda)))
N = space_N(randi(length(space_N)))
z = space_z(randi(length(space_z)))

Tb = T

N_krapiv = N
N_smallw = N
N_er = N

Gs = G

C = 100
