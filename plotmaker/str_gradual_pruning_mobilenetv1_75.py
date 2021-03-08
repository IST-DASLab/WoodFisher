import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

mpl.use('agg')

cmap = plt.cm.Spectral
# cmap = plt.cm.get_cmap('RdBu')
mpl.style.use('seaborn')

fig_dir = './plotmaker/images/'

wf_accs = [72.002, 70.996, 70.726, 70.16, 70.382, 63.226, 69.638, 58.542, 68.166, 64.216, 67.606, 67.278,
           67.718, 68.846, 69.488, 69.65, 69.802, 69.944, 69.99, 70.088]

wf_pts = [0, 4, 8, 8, 12, 12, 16, 16, 20, 20, 24, 24, 30, 40, 50, 60, 70, 80, 90, 99]

mag_accs = [72.002, 71.03, 70.59, 67.982, 70.35, 42.594, 69.744, 17.13, 68.538, 43.732, 67.592, 65.23,
            66.994, 68.702, 69.362, 69.562, 69.686, 69.896, 69.722, 69.768]
mag_pts = [0, 4, 8, 8, 12, 12, 16, 16, 20, 20, 24, 24, 30, 40, 50, 60, 70, 80, 90, 99]

sp_vals = [0, 0.054915698, 0.054915698, 0.39610695, 0.39610695, 0.603059436, 0.603059436, 0.709332551, 0.709332551,
           0.748485692, 0.748485692, 0.754078964, 0.754078964,  0.754078964,  0.754078964,  0.754078964,  0.754078964,  0.754078964,  0.754078964,  0.754078964]

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

plt.annotate("{:.1f}%".format((sp_vals[0] * 100)), (mag_pts[0], mag_accs[0]),
             xytext=(mag_pts[0] - 3.5, mag_accs[0] + 1))
for i, x in enumerate(sp_vals):
    if i % 2 == 1:
        if i == 11:
            x_shift = 1.5
        else:
            x_shift = 0

        plt.annotate("{:.1f}%".format((x * 100)), (mag_pts[i], mag_accs[i]),
                     xytext=(mag_pts[i] - 2.5 + x_shift, mag_accs[i] - 3))
    elif i == (len(sp_vals) - 1):
        continue

plt.legend(loc='lower right')
plt.ylim(bottom=15, top=75)
figpath = os.path.join(fig_dir, 'str_mobilenet_75_grad_fig_final_thin')
plt.savefig(figpath + '.png', format='png', dpi=500, quality=95)
plt.clf()

