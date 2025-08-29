import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# Set Seaborn style
# sns.set(style="darkgrid", context="paper")
# sns.set_style()
sns.axes_style("ticks")
sns.set_context("paper")

sns.color_palette("Paired")

W = (469 / 72) # Figure width in inches, approximately text width
axes_label_font_scale_factor = 2.5 # 3 is too big, 2 is too large.
axes_ticks_font_scale_factor = 2.5
legend_font_scale_factor = 2.0 # perfect, can be reduced to 1.75 even

# Update Matplotlib rcParams
plt.rcParams.update({
    'figure.figsize': (W, W * 3/4), # 4:3 aspect ratio
    'lines.markersize': 6,
    # 'figure.linewidth': 0.5,
    'axes.titlesize': 10 * axes_label_font_scale_factor,
    'axes.labelsize': 10 * axes_label_font_scale_factor,
    'xtick.labelsize': 10 * axes_ticks_font_scale_factor,
    'ytick.labelsize': 10 * axes_ticks_font_scale_factor,
    'legend.fontsize': 10 * legend_font_scale_factor,
    'legend.title_fontsize': 10 * legend_font_scale_factor,
    # 'font.family': 'Arial',
    # 'font.serif': ['Times New Roman'],
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    'axes.grid': True,
    'grid.alpha': 0.75,
    'grid.linestyle': '--',
    'grid.color': 'gray'
})

plt.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts, lmodern}'
plt.rcParams['font.family'] = 'lmodern'

plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['legend.fancybox'] = True
# Set the transparency of the legend frame, the lower more transparent
plt.rcParams['legend.framealpha'] = 0.5  # 0.5 is 50% transparency
plt.rcParams['legend.edgecolor'] = 'black'

matplotlib.rcParams['lines.markersize'] = 10 # 12 is too large

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

print("Seaborn style and Matplotlib rcParams have been set.")