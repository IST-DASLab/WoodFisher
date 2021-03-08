import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

mpl.use('agg')

cmap = plt.cm.Spectral
# cmap = plt.cm.get_cmap('RdBu')
mpl.style.use('seaborn')

fig_dir = './plotmaker/images/'

wf_accs = [72.002, 70.234, 69.422, 68.166, 69.716, 37.262, 66.914, 12.846, 63.08, 33.794, 60.68, 59.53,
           61.64, 62.818, 63.252, 63.56, 63.692, 63.7, 63.866, 63.84]

wf_pts = [0, 4, 8, 8, 12, 12, 16, 16, 20, 20, 24, 24, 36, 46, 56, 66, 76, 86, 96, 99]

mag_accs = [72.002, 70.202, 69.44, 62.576, 69.026, 4.418, 67.242, 0.132, 62.62, 2.262, 59.442, 51.722,
            60.842, 61.93, 62.476, 62.824, 62.996, 63.006, 62.854, 62.94]
mag_pts = [0, 4, 8, 8, 12, 12, 16, 16, 20, 20, 24, 24, 36, 46, 56, 66, 76, 86, 96, 99]

sp_vals = [0, 0.054916407, 0.054916407, 0.46271404, 0.46271404, 0.710067843, 0.710067843, 0.837087235, 0.837087235,
           0.883884002, 0.883884002, 0.890569221, 0.890569221,  0.890569221,  0.890569221,  0.890569221,  0.890569221,  0.890569221,  0.890569221,  0.890569221]

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
            x_shift = 1.5
        else:
            x_shift = 0

        plt.annotate("{:2d}%".format(round(x * 100)), (mag_pts[i], mag_accs[i]),
                     xytext=(mag_pts[i] - 2.5 + x_shift, mag_accs[i] - 3))
    elif i == (len(sp_vals) - 1):
        continue

plt.legend(loc='lower right')
plt.ylim(bottom=-5, top=75)
figpath = os.path.join(fig_dir, 'str_mobilenet_89_grad_fig_final_thin')
plt.savefig(figpath + '.png', format='png', dpi=500, quality=95)
plt.clf()

