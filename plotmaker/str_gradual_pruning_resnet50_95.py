import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

mpl.use('agg')

cmap = plt.cm.Spectral
# cmap = plt.cm.get_cmap('RdBu')
mpl.style.use('seaborn')

fig_dir = './plotmaker/images/'

wf_accs = [77.012, 75.542, 75.022, 74.686, 74.702, 71.63, 74.328, 66.028, 73.632, 55.238, 72.526, 47.966, 70.42, 53.656, 68.452, 64.91,
           69.006, 70.984, 71.6, 71.884, 72.018, 72.158, 72.028]

wf_pts = [0, 1, 6, 6, 11, 11, 16, 16, 21, 21, 26, 26, 31, 31, 36, 36, 46, 56, 66, 76, 86, 92, 99]

mag_accs = [77.012, 75.73, 74.962, 74.27, 74.62, 67.992, 73.81, 51.28, 73.268, 24.7, 72.004, 11.946, 69.694, 19.954, 68.41, 55.068,
            68.938, 70.47, 71.196, 71.328, 71.434, 71.65, 71.544]
mag_pts = [0, 1, 6, 6, 11, 11, 16, 16, 21, 21, 26, 26, 31, 31, 36, 36, 46, 56, 66, 76, 86, 97, 99]

sp_vals = [0, 0.051974657, 0.051974657, 0.355015129, 0.355015129, 0.580838215, 0.580838215, 0.740799354, 0.740799354,
           0.846254145, 0.846254145, 0.908557948, 0.908557948, 0.93906644, 0.93906644, 0.949134983, 0.949134983, 0.949134983, 0.949134983, 0.949134983, 0.949134983, 0.949134983, 0.949134983]

line_colors = cmap(np.linspace(0, 1, 10))
line_color_idx = 0

plt.ylabel('Test accuracy', labelpad=10)
plt.xlabel('Epoch', labelpad=10)

min_y, max_y = 1e100, 1e-100

plt.plot(wf_pts, wf_accs, label='WoodFisher', c=line_colors[9], dash_capstyle='round',
         marker='^', markersize=6, linewidth=1.2)

min_y = min(min(wf_accs), min_y)
max_y = max(max(wf_accs), max_y)
line_color_idx += 1

plt.plot(mag_pts, mag_accs, label='Global Magnitude', c=line_colors[0], dash_capstyle='round',
         marker='o', markersize=6, linewidth=1.2, alpha=0.3)
min_y = min(min(mag_accs), min_y)
max_y = max(max(mag_accs), max_y)

plt.annotate("{:2d}%".format(round(sp_vals[0] * 100)), (mag_pts[0], mag_accs[0]),
             xytext=(mag_pts[0] - 3.5, mag_accs[0] + 1))
for i, x in enumerate(sp_vals):
    if i % 2 == 1:
        if i == 11:
            x_shift = 0
        else:
            x_shift = 0

        plt.annotate("{:2d}%".format(round(x * 100)), (mag_pts[i], mag_accs[i]),
                     xytext=(mag_pts[i] - 3.5 + x_shift, mag_accs[i] - 3))
    elif i == (len(sp_vals) - 1):
        continue

plt.legend(loc='lower right')
plt.ylim(bottom=-5, top=80)
figpath = os.path.join(fig_dir, 'str_imagenet_95_grad_fig_final_thin')
plt.savefig(figpath + '.png', format='png', dpi=500, quality=95)
plt.clf()

