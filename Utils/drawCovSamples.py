import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
# matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

def plot_cov2D_samples(cov, sample_num = 500,  **kwargs):
    assert cov.shape[0] == cov.shape[1] == 2
    mu, sigma = np.array([0, 0], dtype=np.float), cov
    sample_generated = np.random.multivariate_normal(mu, sigma, sample_num)
    sampleplot, = plt.plot(sample_generated[:, 0], sample_generated[:, 1], '.', **kwargs)
    return sampleplot


"""
    cov : The 2x2 covariance matrix to base the ellipse on
    pos : The location of the center of the ellipse. Expects a 2-element
        sequence of [x0, y0].
    nstd : The radius of the ellipse in numbers of standard deviations.
        Defaults to 2 standard deviations.
    ax : The axis that the ellipse will be plotted on. Defaults to the 
        current axis.
"""


def plot_cov_ellipse(cov, nstd=2, ax=None, **kwargs):

    def sort_eig(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()
    pos = (0, 0)
    vals, vecs = sort_eig(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    # limit bounding box
    theta_rad = np.deg2rad(theta)
    wid1 = width * abs(np.cos(theta_rad)) + height * abs(np.sin(theta_rad))
    wid2 = width * abs(np.sin(theta_rad)) + height * abs(np.cos(theta_rad))
    maxwidth = np.max([wid1 * 1.2, wid2 * 1.2])
    ax.add_artist(ellip)
    return ellip, maxwidth / 2


def plot_cov2d(cov, title = None, fignew=False):
    tickfontsize=15
    if fignew:
        fig = plt.figure()
    else:
        fig = plt.gca().figure
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False

    kwrg_oval = {'edgecolor': 'k', 'linewidth': 0.6, 'facecolor': (35/255,235/255,186/255,0.8)}
    kwrg_pts ={'mew':0, 'markerfacecolor':(0.996,0.263,0.396,1)}

    #anti color
    # kwrg_oval = {'edgecolor': 'k', 'linewidth': 0.6, 'facecolor': (0.996,0.263,0.396,0.8)}
    # kwrg_pts ={'mew':0, 'markerfacecolor':(35/255,235/255,186/255,1)}

    samps = plot_cov2D_samples(cov, **kwrg_pts)
    o_eclip, w = plot_cov_ellipse(cov, nstd=2, **kwrg_oval)
    plt.axis('equal')
    plt.xlim([-w, w])
    plt.ylim([-w, w])
    plt.title(title)
    axes = plt.gca()
    # set the bounding box: equal width and height
    axes.set_aspect('equal', 'box')
    # plt.close()
    ax = fig.gca()
    bwith = 2
    ax.ticklabel_format(style='sci')
    ax.ticklabel_format(scilimits=(0, 0))
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.locator_params(axis='y', nbins=2)
    plt.locator_params(axis='x', nbins=2)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(tickfontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(tickfontsize)
    return fig

if __name__ == '__main__':
    cov = np.array([[10, -12], [-12, 30]])
    plot_cov2d(cov)
    plt.show()
    pass
