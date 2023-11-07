from symm_group_rep import *
from matplotlib import pyplot as plt
import numpy as np
from itertools import combinations


print(dim_S_lbd(Partition([6,4,1,1])))

## Example computations for function approximation
N = 20
max_dim_list = []
min_dim_list = []
max_p_list = []
for n in range(N):
    p_list = Partition.generate_all_partitions(n+1)
    dim_list = [dim_S_lbd(p) for p in p_list]
    max_dim_list.append(max(dim_list))
    #print(min(dim_list))
    max_p_list.append(p_list[dim_list.index(max(dim_list))])

symm_p_list = []
p_list = Partition.generate_all_partitions(11)
for p in p_list:
    if p == p.conjugate:
        symm_p_list.append(p)
        #print(p, dim_S_lbd(p))


n_list = np.array(range(1, N+1))
f_list_log = 1.7320 * n_list - 18
f_list = np.exp(f_list_log)

max_dim_list = np.array(max_dim_list)
max_dim_list_log = np.log(max_dim_list)
fig, ax = plt.subplots(1,3)
ax[0].plot(n_list, max_dim_list, label="$\dim \max_{\lambda\in P_n}{S^\lambda}$")
ax[0].plot(n_list, f_list, label="r$\exp{\alpha n+\beta}$")
ax[0].legend()
ax[1].plot(n_list, max_dim_list_log, label="$\log \dim \max_{\lambda\in P_n}S^\lambda$")
ax[1].plot(n_list, f_list_log, label=r"$\alpha n+\beta$")
ax[1].legend()
ax[2].plot(n_list, f_list_log/n_list, label=r"$\frac{\log \dim \max_{\lambda\in P_n}S^\lambda}{n}$")
ax[2].legend()
plt.show()
print((max_dim_list[-1]-max_dim_list[-3])/2)

t_list = Tabloid.generate_all_tabloids([4,3,1])
# print(t_list)
t0 = t_list[0]
t_orbit = []
flag = 1
# print(t_list[3],t_list[10])

tabloids = [Tabloid(t) for t in t_list]
t_max = max(tabloids)
t_min = min(tabloids)
print(f"max element is {t_max} \nmin element is {t_min}") 