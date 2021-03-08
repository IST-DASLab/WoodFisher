import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

mpl.use('agg')

cmap = plt.cm.Spectral
# cmap = plt.cm.get_cmap('RdBu')
mpl.style.use('seaborn')

fig_dir = './plotmaker/images/'

wf_accs = [77.012, 75.638, 75.134, 74.832, 74.676, 72.656, 74.526, 69.264, 74.038, 65.142, 73.29, 65.072, 72.108, 68.3, 71.438, 70.45,
           72.488, 74.374, 74.82, 74.966, 75.118, 75.26, 75.09]

wf_pts = [0, 1, 6, 6, 11, 11, 16, 16, 21, 21, 26, 26, 31, 31, 36, 36, 46, 56, 66, 76, 86, 97, 99]

mag_accs = [77.012, 75.622, 75.004, 74.338, 75.028, 69.696, 74.5, 63.298, 73.482, 50.48, 72.56, 47.772, 71.804, 59.662, 71.456, 69.164,
            72.424, 74.114, 74.706, 74.854, 74.914, 75.094, 75.004]
mag_pts = [0, 1, 6, 6, 11, 11, 16, 16, 21, 21, 26, 26, 31, 31, 36, 36, 46, 56, 66, 76, 86, 95, 99]

sp_vals = [0, 0.051974657, 0.051974657, 0.338179573, 0.338179573, 0.551456893, 0.551456893, 0.702531442, 0.702531442,
           0.802127498, 0.802127498, 0.860969966, 0.860969966, 0.889783516, 0.889783516, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90]

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
                     xytext=(mag_pts[i] - 3.5 + x_shift, mag_accs[i] - 2))
    elif i == (len(sp_vals) - 1):
        continue

plt.legend(loc='lower right')
plt.ylim(bottom=45, top=80)
figpath = os.path.join(fig_dir, 'str_imagenet_90_grad_fig_final_thin')
plt.savefig(figpath + '.png', format='png', dpi=500, quality=95)
plt.clf()

