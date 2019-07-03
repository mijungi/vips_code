import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

# load our results
acc_ours = np.load('accuracy_ours.npy')
# non-private, then private with sigma = 0, 1, 10, 50

non_private_ours = acc_ours[0,:]
ours_sigma1 = acc_ours[1,:]
ours_sigma10 = acc_ours[2,:]
ours_sigma50 = acc_ours[3,:]

# load DPVI
acc_dpvi = np.load('accuracy_advi.npy')
dpvi_sigma1 = acc_dpvi[0,:]
dpvi_sigma10 = acc_dpvi[1,:]
dpvi_sigma50 = acc_dpvi[2,:]

non_private_dvi = np.load('accuracy_dvi.npy')

####

random_dists = ['VIPS(sig=0)', 'VIPS(sig=1)', 'VIPS(sig=10)', 'VIPS(sig=50)',
                'DPVI(sig=0)', 'DPVI(sig=1)', 'DPVI(sig=10)', 'DPVI(sig=50)']
N = 20

data = [
    non_private_ours,
    ours_sigma1,
    ours_sigma10,
    ours_sigma50,
    non_private_dvi,
    dpvi_sigma1,
    dpvi_sigma10,
    dpvi_sigma50,
]

fig, ax1 = plt.subplots(figsize=(10, 6))
fig.canvas.set_window_title('Adult data')
fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='red', marker='+')

# Add a horizontal grid to the plot, but make it very light in color
# so we can use it for reading data values but not be distracting
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)

# Hide these grid behind plot objects
ax1.set_axisbelow(True)
ax1.set_title('Comparison of VIPS to DPVI')
ax1.set_xlabel('Different noise level')
ax1.set_ylabel('AUC on test data')

# Now fill the boxes with desired colors
box_colors = ['darkkhaki', 'royalblue']
num_boxes = len(data)
medians = np.empty(num_boxes)
for i in range(num_boxes):
    box = bp['boxes'][i]
    boxX = []
    boxY = []
    for j in range(4):
        boxX.append(box.get_xdata()[j])
        boxY.append(box.get_ydata()[j])
    box_coords = np.column_stack([boxX, boxY])
    # Alternate between Dark Khaki and Royal Blue
    if i<4:
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[1]))
    else:
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[0]))
    # Now draw the median lines back over what we just filled in
    med = bp['medians'][i]
    medianX = []
    medianY = []
    # for j in range(2):
    for j in range(2):
        medianX.append(med.get_xdata()[j])
        medianY.append(med.get_ydata()[j])
        ax1.plot(medianX, medianY, 'k')
    medians[i] = medianY[0]
    # Finally, overplot the sample averages, with horizontal alignment
    # in the center of each box
    ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
             color='w', marker='*', markeredgecolor='k')

# Set the axes ranges and axes labels
ax1.set_xlim(0.5, num_boxes + 0.5)
top = 1
bottom = 0
ax1.set_ylim(bottom, top)
ax1.set_xticklabels(np.repeat(random_dists, 1),
                    rotation=45, fontsize=8)

# Due to the Y-axis scale being different across samples, it can be
# hard to compare differences in medians across the samples. Add upper
# X-axis tick labels with the sample medians to aid in comparison
# (just use two decimal places of precision)
pos = np.arange(num_boxes) + 1
upper_labels = [str(np.round(s, 2)) for s in medians]
# upper_labels = [str(np.round(s, 1)) for s in medians]
weights = ['bold', 'semibold']
for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
    # k = tick % 2
    # k = tick
    if tick<4:
        ax1.text(pos[tick], .95, upper_labels[tick],
             transform=ax1.get_xaxis_transform(),
             horizontalalignment='center', size='x-small',
             weight=weights[1], color=box_colors[1])
    else:
        ax1.text(pos[tick], .95, upper_labels[tick],
             transform=ax1.get_xaxis_transform(),
             horizontalalignment='center', size='x-small',
             weight=weights[1], color=box_colors[0])

# Finally, add a basic legend
# fig.text(0.80, 0.08, 'average over 20 random initalizations',
#          backgroundcolor=box_colors[0], color='black', weight='roman',
#          size='x-small')
# fig.text(0.80, 0.045, 'IID Bootstrap Resample',
#          backgroundcolor=box_colors[1],
#          color='white', weight='roman', size='x-small')
fig.text(0.80, 0.08, '*', color='white', backgroundcolor='silver',
         weight='roman', size='medium')
fig.text(0.815, 0.085, ' Average Value', color='black', weight='roman',
         size='x-small')

plt.show()