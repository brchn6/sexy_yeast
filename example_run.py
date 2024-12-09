import cmn

# Size of genome
N = 1000
# Sparsity param 0<=rho<=1
rho = 0.25
# Epistatic strength param 0<=beta<=1
beta = 0.5

sig_0 = cmn.init_sigma(N)
h = cmn.init_h(N, beta)
J = cmn.init_J(N, beta, rho)
F_off = cmn.calc_F_off(sig_0, h, J)
init_fit = cmn.compute_fit_slow(sig_0, h, J, F_off)
# init_fit should be 1 of course, as seen in the documentation

flip_seq = cmn.relax_sk(sig_0, h, J)
# This is the heart of the project, it runs an optimization scheme and return an array
# [k_1, k_2, ..., k_n] where k_i is the index of the spin flipped at step i
# From this we can recreate the sigma vector at any time we want.
# Thi is, of course, all in the sswm regime.

num_of_muts = len(flip_seq)
sig_final = cmn.compute_sigma_from_hist(sig_0, flip_seq, num_of_muts)
final_fit = cmn.compute_fit_slow(sig_final, h, J, F_off)
dfe = cmn.calc_dfe(sig_final, h, J)