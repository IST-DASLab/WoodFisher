import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

mpl.use('agg')

cmap = plt.cm.Spectral
mpl.style.use('seaborn')

fig_dir = './plotmaker/'

lobs_acc = [92.25, 92.35, 92.25, 92.15, 91.25, 89.9, 82.75, 63.75]
lobs_loose_acc = [92.35, 92.4, 92.35, 92.25, 91.25, 89.9, 83, 64]

wf_indep_acc = [92.862, 93.002, 92.942, 92.712, 92.162, 90.574, 85.078, 67.13]
wf_joint_acc = [92.862, 92.966, 92.948, 92.78, 92.538, 91.726, 89.09, 84.806]

pts = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65]
pts = [x*100 for x in pts]

line_colors = cmap(np.linspace(0, 1, 10))
line_color_idx = 0
plt.clf()
plt.ylabel('Test accuracy (Top-5)', labelpad=10)
plt.xlabel('Sparsity', labelpad=10)

min_y, max_y = 1e100, 1e-100

plt.plot(pts, lobs_acc, label='Layerwise Optimal Brain Surgeon', c=line_colors[1], dash_capstyle='round',
         marker='o', markersize=6, alpha=0.5)

min_y = min(min(lobs_acc), min_y)
max_y = max(max(lobs_acc), max_y)
line_color_idx += 1

plt.plot(pts, wf_indep_acc, label='WoodFisher (layerwise)', c=line_colors[0], dash_capstyle='round',
         marker='^', markersize=6,  linestyle='dashdot', alpha=0.9)

min_y = min(min(wf_indep_acc), min_y)
max_y = max(max(wf_indep_acc), max_y)
line_color_idx += 1

plt.plot(pts, wf_joint_acc, label='WoodFisher (global)', c=line_colors[9], dash_capstyle='round',
         marker='s', markersize=6, linestyle='dashed', alpha=0.8)

min_y = min(min(wf_joint_acc), min_y)
max_y = max(max(wf_joint_acc), max_y)
line_color_idx += 1

plt.legend(loc='lower left')
plt.ylim(bottom=min_y-4.5, top=max_y+1.5)
plt.xlim(left=-2, right=75)
plt.subplots_adjust(top=0.8)
figpath = os.path.join(fig_dir, 'lobs_comparison_final')
plt.savefig(figpath + '.png', format='png', dpi=500, quality=95)
plt.clf()