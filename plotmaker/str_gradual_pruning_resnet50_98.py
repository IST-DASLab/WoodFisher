import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

mpl.use('agg')

cmap = plt.cm.Spectral
# cmap = plt.cm.get_cmap('RdBu')
mpl.style.use('seaborn')

fig_dir = './plotmaker/images/'

wf_accs = [77.012, 75.626, 75.17, 74.81, 74.984, 71.276, 74.532, 61.326, 73.338, 40.042, 71.644, 15.606, 67.644, 11.536, 62.714, 43.768,
           62.286, 64.25, 64.736, 65.062, 65.326, 65.466, 65.232]

wf_pts = [0, 1, 6, 6, 11, 11, 16, 16, 21, 21, 26, 26, 31, 31, 36, 36, 46, 56, 66, 76, 87, 95, 99]

mag_accs = [77.012, 75.864, 75.106, 74.212, 74.566, 66.358, 73.89, 45.078, 73.096, 9.6, 71.084, 0.822, 67.55, 0.32, 62.054, 4.534,
            61.628, 62.868, 63.624, 63.772, 64.166, 63.99, 64.02]
mag_pts = [0, 1, 6, 6, 11, 11, 16, 16, 21, 21, 26, 26, 31, 31, 36, 36, 46, 56, 66, 76, 87, 95, 99]

sp_vals = [0, 0.051974657, 0.051974657, 0.365116463, 0.365116463, 0.598466969, 0.598466969, 0.763760196, 0.763760196,
           0.872730086, 0.872730086, 0.937110738, 0.937110738, 0.968636133, 0.968636133, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98]

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
figpath = os.path.join(fig_dir, 'str_imagenet_98_grad_fig_final_thin')
plt.savefig(figpath + '.png', format='png', dpi=500, quality=95)
plt.clf()

