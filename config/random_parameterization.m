space_L = [3 4 5 6 7];
space_K = [25 50 100 200 400 800 1600];
space_T = [10 25 50 100];
space_B = [5 10 20];
space_G = [1 5 10 20];
space_alpha = [1.0 0.1 0.01 0.001];
space_lambda = [0.1 0.01 0.001 0.0001];
space_N = [100 250 500];
space_z = [10 50 100 200];

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
