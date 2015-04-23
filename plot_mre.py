#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt


n_groups = 5
ucsd_mre = (7.25, 7.15, 7, 7.08, 7.08)
pets_mre = (20.41, 20.23, 18.12, 17.89, 18.43)
mall_mre = (9.98, 9.95, 9.9, 9.9, 9.94)
avg_mae = np.mean((ucsd_mre, pets_mre, mall_mre), axis=0)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.25

opacity = 0.7
error_config = {'ecolor': '0.3'}

plt.bar(index, ucsd_mre, bar_width,
        alpha=opacity,
        color='b',
        label='UCSD')
plt.bar(index + bar_width, pets_mre, bar_width,
        alpha=opacity,
        color='g',
        label='PETS')
plt.bar(index + 2 * bar_width, mall_mre, bar_width,
        alpha=opacity,
        color='r',
        label='MALL')
plt.plot(index + 1.5 * bar_width, avg_mae, 'o-', label="Average")

ylim_u = 23
ylim_l = 6
plt.ylim(ylim_l, ylim_u)

plt.xlabel(u"Regression Models")
plt.ylabel('MAE')
plt.title('MAE of different models')
plt.xticks(index + bar_width * 1.5, ('LR', 'RR', 'L-SVR', 'RBF-SVR', 'GPR'))
plt.yticks(np.linspace(ylim_l, ylim_u, 18))
plt.legend(loc ='best')
plt.gca().yaxis.grid(True)
# ax = plt.gca()
# ax.grid(True)
plt.tight_layout()
plt.show()
