#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt


n_groups = 5
ucsd_mae = (1.67, 1.66, 1.62, 1.62, 1.63)
pets_mae = (3.48, 3.47, 3.35, 3.35, 3.36)
mall_mae = (2.86, 2.85, 2.85, 2.79, 2.85)
avg_mae = np.mean((ucsd_mae, pets_mae, mall_mae), axis=0)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.25

opacity = 0.7
error_config = {'ecolor': '0.3'}

plt.bar(index, ucsd_mae, bar_width,
        alpha=opacity,
        color='b',
        label='UCSD')
plt.bar(index + bar_width, pets_mae, bar_width,
        alpha=opacity,
        color='g',
        label='PETS')
plt.bar(index + 2 * bar_width, mall_mae, bar_width,
        alpha=opacity,
        color='r',
        label='MALL')
plt.plot(index + 1.5 * bar_width, avg_mae, 'o-', label="Average")

ylim_u = 4
plt.ylim(1.5, ylim_u)

plt.xlabel(u"Regression Models")
plt.ylabel('MAE')
plt.title('MAE of different models')
plt.xticks(index + bar_width * 1.5, ('LR', 'RR', 'L-SVR', 'RBF-SVR', 'GPR'))
plt.yticks(np.linspace(1.5, ylim_u, 11))
plt.legend(loc ='best')
plt.gca().yaxis.grid(True)
# ax = plt.gca()
# ax.grid(True)
plt.tight_layout()
plt.show()
