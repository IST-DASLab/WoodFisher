import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

mpl.use('agg')

cmap = plt.cm.Spectral
# cmap = plt.cm.get_cmap('RdBu')
mpl.style.use('seaborn')

fig_dir = './plotmaker/images/'

wf_accs = [77.012, 76.442, 75.998, 75.748, 75.726, 74.02, 75.388, 72.308, 74.984, 71.27, 74.542, 71.758, 74.146, 72.944, 73.666, 73.396,
           74.684, 76.152, 76.456, 76.496, 76.664, 76.732, 76.632]

wf_pts = [0, 1, 6, 6, 11, 11, 16, 16, 21, 21, 26, 26, 31, 31, 36, 36, 46, 56, 66, 76, 86, 92, 99]

mag_accs = [77.012, 76.404, 75.908, 75.458, 75.636, 72.754, 75.102, 67.536, 75.192, 66.438, 74.452, 67.074, 73.862, 70.514, 73.746, 73.254,
            74.592, 75.964, 76.23, 76.394, 76.52, 76.596, 76.52]
mag_pts = [0, 1, 6, 6, 11, 11, 16, 16, 21, 21, 26, 26, 31, 31, 36, 36, 46, 56, 66, 76, 86, 91, 99]

sp_vals = [0, 0.051974657, 0.051974657, 0.304508384, 0.304508384, 0.492694288, 0.492694288, 0.625995225, 0.625995225,
           0.713874204, 0.713874204, 0.765794079, 0.765794079, 0.791217823, 0.791217823, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80,0.80]

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
            x_shift = 3
        else:
            x_shift = 0

        plt.annotate("{:2d}%".format(round(x * 100)), (mag_pts[i], mag_accs[i]),
                     xytext=(mag_pts[i] - 3.5 + x_shift, mag_accs[i] - 1))
    elif i == (len(sp_vals) - 1):
        continue

plt.legend(loc='lower right')
plt.ylim(bottom=55, top=80)
figpath = os.path.join(fig_dir, 'str_imagenet_80_grad_fig_final_thin')
plt.savefig(figpath + '.png', format='png', dpi=500, quality=95)
plt.clf()

