import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from contextlib import contextmanager

from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

data_color = "black"
fit_color = "xkcd:azure"
sig_color = "xkcd:coral"
bck_color = "xkcd:teal"
diff_color = "xkcd:red"

@contextmanager
def plot(name, prefix) : 
    """
    Helper context to open matplotlib canvas and then save it to pdf and png files
    """
    fig, ax = plt.subplots(figsize = (4, 3) )
    fig.subplots_adjust(bottom=0.15, left = 0.20, right = 0.95, top = 0.98)
    yield fig, ax
    fig.savefig(prefix + name + ".pdf")
    fig.savefig(prefix + name + ".png")


def set_lhcb_style(grid=True, size=10, usetex="auto", font="serif"):
    """
    Set matplotlib plotting style close to "official" LHCb style
    (serif fonts, tick sizes and location, etc.)
      :param grid:   Enable grid on the plots
      :param size:   Font size
      :param usetex: Use LaTeX for labels
      :param font:   Font family
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    if usetex == "auto":
        plt.rc("text", usetex=os.path.isfile("/usr/bin/latex"))
    else:
        plt.rc("text", usetex=usetex)
    plt.rc("font", family=font, size=size)
    plt.rcParams["axes.linewidth"] = 1.3
    plt.rcParams["axes.grid"] = grid
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["axes.axisbelow"] = False
    plt.rcParams["xtick.major.width"] = 1
    plt.rcParams["ytick.major.width"] = 1
    plt.rcParams["xtick.minor.width"] = 1
    plt.rcParams["ytick.minor.width"] = 1
    plt.rcParams["xtick.major.size"] = 6
    plt.rcParams["ytick.major.size"] = 6
    plt.rcParams["xtick.minor.size"] = 3
    plt.rcParams["ytick.minor.size"] = 3
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.left"] = True
    plt.rcParams["ytick.right"] = True


def label_title(title, units=None):
    label = title
    if units:
        title += " [" + units + "]"
    return title


def y_label_title(range, bins, units=None):
    binw = (range[1] - range[0]) / bins
    if units == None:
        title = f"Entries / {binw}"
    else:
        title = f"Entries / ({binw:g} {units})"
    return title


def plot_distr2d(
    xarr,
    yarr,
    bins,
    ranges,
    fig,
    ax,
    labels,
    cmap="YlOrBr",
    log=False,
    ztitle=None,
    title=None,
    units=(None, None),
    weights=None,
    colorbar=True,
):
    """
    Plot 2D distribution including colorbox.
      :param xarr:   array of x-coordinates
      :param yarr:   array of y-coordinates
      :param bins:   2-element tuple with number of bins in x and y axes, e.g. (50, 50)
      :param ranges: 2-element tuple of ranges in x and y dimensions, e.g. ((0, 1), (-1, 1))
      :param fig:    matplotlib figure object
      :param ax:     matplotlib axis object
      :param labels: Axis label titles (2-element tuple)
      :param cmap:   matplotlib colormap name
      :param log:    if True, use log z scale
      :param ztitle: z axis title (if None, use "Entries")
      :param title:  plot title
      :param units:  2-element tuple of x axis and y axis units
    """

    def fasthist2d(xvals, yvals, bins, ranges, weights):
        vals = (np.array(xvals), np.array(yvals))
        cuts = (
            (vals[0] >= ranges[0][0])
            & (vals[0] < ranges[0][1])
            & (vals[1] >= ranges[1][0])
            & (vals[1] < ranges[1][1])
        )
        if weights is None : _weights = None
        else : _weights = weights[cuts]
        c = (
            (vals[0][cuts] - ranges[0][0]) / (ranges[0][1] - ranges[0][0]) * bins[0]
        ).astype(np.int_)
        c += bins[0] * (
            (vals[1][cuts] - ranges[1][0]) / (ranges[1][1] - ranges[1][0]) * bins[1]
        ).astype(np.int_)
        H = np.bincount(c, minlength=bins[0] * bins[1], weights=_weights)[
            : bins[0] * bins[1]
        ].reshape(bins[1], bins[0])
        return (
            H,
            np.linspace(ranges[0][0], ranges[0][1], bins[0] + 1),
            np.linspace(ranges[1][0], ranges[1][1], bins[1] + 1),
        )

    counts, xedges, yedges = fasthist2d(
        xarr, yarr, bins=bins, ranges=ranges, weights=weights
    )

    norm = None
    vmax = np.max(counts)
    vmin = np.min(counts)
    if len(ranges) > 2:
        vmin = ranges[2][0]
        vmax = ranges[2][1]
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    if log:
        if vmin <= 0.0:
            vmin = 0.3
        if vmax <= vmin:
            vmax = vmin
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)

    X, Y = np.meshgrid(xedges, yedges)
    p = ax.pcolormesh(X, Y, counts, cmap=cmap, norm=norm, linewidth=0, rasterized=True)
    ax.set_xlabel(label_title(labels[0], units[0]), ha="right", x=1.0)
    ax.set_ylabel(label_title(labels[1], units[1]), ha="right", y=1.0)
    if title:
        ax.set_title(title)
    zt = ztitle
    if not ztitle:
        zt = r"Entries"
    jd = { zt : list(zip(((X[1:,1:]+X[:-1,:-1])/2.).flatten(), ((Y[1:,1:]+Y[:-1,:-1])/2.).flatten(), counts.flatten())) }
    if colorbar:
        cb = fig.colorbar(p, pad=0.01, ax=ax)
        cb.ax.set_ylabel(zt, ha="right", y=1.0)
        if log:
            cb.ax.set_yscale("log")
    return (vmin, vmax), jd


def plot_distr1d(
    arr,
    bins,
    range,
    ax,
    label,
    log=False,
    units=None,
    weights=None,
    title=None,
    color=None,
    legend=None,
    errors=False,
    normalise=False, 
    fill=True, 
    line=False, 
    legend_ax=None, 
):
    """
    Plot 1D histogram from the data in array.
      :param arr:    array of x-coordinates
      :param bins:   number of bins in x axis
      :param range:  ranges in x dimension, e.g. (0, 1)
      :param ax:     matplotlib axis object
      :param label:  x-axis label title
      :param log:    if True, use log y scale
      :param ztitle: z axis title (if None, use "Entries")
      :param title:  plot title
      :param units:  2-element tuple of x axis and y axis units
    """
    jd = {}
    if isinstance(weights, list):
        xarr = None
        for i, w in enumerate(weights):
            hist, edges = np.histogram(arr, bins=bins, range=range, weights=w)
            left, right = edges[:-1], edges[1:]
            if xarr is None:
                xarr = np.array([left, right]).T.flatten()
            dataarr = np.array([hist, hist]).T.flatten()
            if color:
                this_color = color[i]
            else:
                this_color = f"C{i}"
            if legend:
                lab = legend[i]
            else:
                lab = None
            ax.plot(xarr, dataarr, color=this_color, label=lab)
            jd[lab] = list(zip(0.5*(left+right), hist.astype(float)))
            if fill : ax.fill_between(xarr, dataarr, 0.0, color=this_color, alpha=0.1)
    elif isinstance(arr, list):
        xarr = None
        for i, a in enumerate(arr):
            hist, edges = np.histogram(a, bins=bins, range=range, weights=weights)
            if normalise : 
                hist = hist.astype(np.float64)*float(bins)/np.sum(hist)/(range[1]-range[0])
            left, right = edges[:-1], edges[1:]
            if xarr is None:
                if line : 
                    xarr = (left+right)/2.
                else : 
                    xarr = np.array([left, right]).T.flatten()
            if line : 
                dataarr = hist
            else : 
                dataarr = np.array([hist, hist]).T.flatten()
            if color:
                this_color = color[i]
            else:
                this_color = f"C{i}"
            if legend:
                lab = legend[i]
            else:
                lab = None
            ax.plot(xarr, dataarr, color=this_color, label=lab)
            jd[lab] = list(zip(0.5*(left+right), hist.astype(float)))
            if fill : ax.fill_between(xarr, dataarr, 0.0, color=this_color, alpha=0.1)
    else:
        if color:
            this_color = color
        else:
            this_color = data_color
        hist, edges = np.histogram(arr, bins=bins, range=range, weights=weights)
        left, right = edges[:-1], edges[1:]
        xarr = np.array([left, right]).T.flatten()
        dataarr = np.array([hist, hist]).T.flatten()
        if errors:
            hist2, _ = np.histogram(arr, bins=bins, range=range, weights= None if weights is None else weights**2)
            xarr = (left + right) / 2.0
            ax.errorbar(
                xarr, hist, np.sqrt(hist2), color=this_color, marker=".", linestyle=""
            )
            jd[label] = list(zip(0.5*(left+right), hist.astype(float), np.sqrt(hist2)))
        else:
            ax.plot(xarr, dataarr, color=this_color, label = legend)
            if fill : ax.fill_between(xarr, dataarr, 0.0, color=this_color, alpha=0.1)
            jd[label] = list(zip(0.5*(left+right), hist.astype(float)))
    if not log:
        ax.set_ylim(bottom=0.0)
    else:
        ax.set_ylim(bottom=0.1)
        ax.set_yscale("log")
    ax.set_xlabel(label_title(label, units), ha="right", x=1.0)
    ax.set_ylabel(y_label_title(range, bins, units), ha="right", y=1.0)
    if title is None:
        ax.set_title(label + r" distribution")
    elif title:
        ax.set_title(title)
    if legend:
        if legend_ax:
            h, l = ax.get_legend_handles_labels()
            legend_ax.legend(h, l, borderaxespad=0)
            legend_ax.axis("off")
        else:
            ax.legend(loc="best")
    return jd


def plot_distr1d_comparison(
    data,
    fit,
    bins,
    range,
    ax,
    label,
    log=False,
    units=None,
    weights=None,
    pull=False,
    cweights=None,
    dataweights=None,
    title=None,
    legend=None,
    color=None,
    data_alpha=1.0,
    legend_ax=None,
    scale=None, 
):
    """
    Plot 1D histogram and its fit result.
      hist : histogram to be plotted
      func : fitting function in the same format as fitting.fit_hist1d
      pars : list of fitted parameter values (output of fitting.fit_hist2d)
      ax   : matplotlib axis object
      label : x axis label title
      units : Units for x axis
    """
    if not legend == False:
        if legend == None : 
          dlab, flab = "Data", "Fit"
        elif cweights is None and len(legend) == 2 : 
          dlab, flab = legend
        elif (cweights is not None) and (len(legend) == len(cweights)+2) : 
          dlab, flab = legend[-2:]
        else : 
          dlab, flab = "Data", "Fit"
    else:
        dlab, flab = None, None
    datahist, _ = np.histogram(data, bins=bins, range=range, weights=dataweights)
    fithist1, edges = np.histogram(fit, bins=bins, range=range, weights=weights)
    fitscale = scale if scale is not None else np.sum(datahist) / np.sum(fithist1) 
    fithist = fithist1 * fitscale
    left, right = edges[:-1], edges[1:]
    fitarr = np.array([fithist, fithist]).T.flatten()
    dataarr = np.array([datahist, datahist]).T.flatten()
    xarr = np.array([left, right]).T.flatten()
    ax.plot(xarr, fitarr, label=flab, color=fit_color)

    if isinstance(cweights, list):
        cxarr = None
        for i, w in enumerate(cweights):
            if weights is not None :
                w2 = w * weights
            else:
                w2 = w
            chist, cedges = np.histogram(fit, bins=bins, range=range, weights=w2)
            if cxarr is None:
                cleft, cright = cedges[:-1], cedges[1:]
                cxarr = (cleft + cright) / 2.0
            fitarr = chist * fitscale
            if color:
                this_color = color[i]
            else:
                this_color = f"C{i+1}"
            if legend:
                lab = legend[i]
            else:
                lab = None
            ax.plot(cxarr, fitarr, color=this_color, label=lab)
            ax.fill_between(cxarr, fitarr, 0.0, color=this_color, alpha=0.1)

    xarr = (left + right) / 2.0
    datahist2, _ = np.histogram(data, bins=bins, range=range, weights = None if dataweights is None else dataweights**2)
    ax.errorbar(
        xarr,
        datahist,
        np.sqrt(datahist2),
        label=dlab,
        color=data_color,
        marker=".",
        linestyle="",
        alpha=data_alpha,
    )

    if not legend == False:
        if legend_ax:
            h, l = ax.get_legend_handles_labels()
            legend_ax.legend(h, l, borderaxespad=0)
            legend_ax.axis("off")
        else:
            ax.legend(loc="best")
    if not log:
        ax.set_ylim(bottom=0.0)
    else:
        ax.set_ylim(bottom=0.1)
        ax.set_yscale("log")
    ax.set_xlabel(label_title(label, units), ha="right", x=1.0)
    ax.set_ylabel(y_label_title(range, bins, units), ha="right", y=1.0)
    if title is None:
        ax.set_title(label + r" distribution")
    elif title:
        ax.set_title(title)
    if pull:
        xarr = np.array([left, right]).T.flatten()
        with np.errstate(divide="ignore", invalid="ignore"):
            pullhist = (datahist - fithist) / np.sqrt(datahist)
            pullhist[datahist == 0] = 0
        # pullhist = np.divide(datahist-fithist, np.sqrt(datahist), out=np.zeros_like(datahist), where=(datahist>0) )
        pullarr = np.array([pullhist, pullhist]).T.flatten()
        ax2 = ax.twinx()
        ax2.set_ylim(bottom=-10.0)
        ax2.set_ylim(top=10.0)
        ax2.plot(xarr, pullarr, color=diff_color, alpha=0.3)
        ax2.grid(False)
        ax2.set_ylabel(r"Pull", ha="right", y=1.0)
        return [ax2]
    return []


def plot_distr_comparison(
    arr1, arr2, bins, ranges, labels, fig, axes, units=None, cmap="jet"
):
    dim = arr1.shape[1]

    for i in range(dim):
        plot_distr1d_comparison(
            arr1[:, i], arr2[:, i], bins[i], ranges[i], axes[0, i], labels[i], pull=True
        )

    n = 0
    for i in range(dim):
        for j in range(i):
            if dim % 2 == 0:
                ax1 = axes[n // (dim // 2) + 1, n % (dim // 2)]
                ax2 = axes[n // (dim // 2) + 1, n % (dim // 2) + 1]
            else:
                ax1 = axes[2 * (n // dim) + 1, n % dim]
                ax2 = axes[2 * (n // dim) + 2, n % dim]
            plot_distr2d(
                arr1[:, i],
                arr1[:, j],
                bins=(bins[i], bins[j]),
                ranges=(ranges[i], ranges[j]),
                fig=fig,
                ax=ax1,
                labels=(labels[i], labels[j]),
                cmap=cmap,
            )
            plot_distr2d(
                arr2[:, i],
                arr2[:, j],
                bins=(bins[i], bins[j]),
                ranges=(ranges[i], ranges[j]),
                fig=fig,
                ax=ax2,
                labels=(labels[i], labels[j]),
                cmap=cmap,
            )
            n += 1


class MultidimDisplay:
    def __init__(
        self, data, norm, bins, ranges, labels, fig, axes, units=None, cmap="jet", dataweights=None
    ):
        self.dim = data.shape[1]
        self.data = data
        self.norm = norm
        self.bins = bins
        self.dataweights = dataweights
        self.ranges = ranges
        self.labels = labels
        self.fig = fig
        self.axes = axes
        self.units = units
        self.cmap = cmap
        self.size = data.shape[0]
        self.first = True
        self.newaxes = []
        self.zrange = {}
        n = 0
        for i in range(self.dim):
            for j in range(i):
                if self.dim % 2 == 0:
                    ax1 = axes[(n // (self.dim // 2)) + 1, 2 * (n % (self.dim // 2))]
                else:
                    ax1 = axes[2 * (n // self.dim) + 1, n % self.dim]
                self.zrange[(i, j)], _ = plot_distr2d(
                    data[:, i],
                    data[:, j],
                    bins=(bins[i], bins[j]),
                    ranges=(ranges[i], ranges[j]),
                    fig=fig,
                    ax=ax1,
                    labels=(labels[i], labels[j]),
                    cmap=cmap,
                    weights=dataweights,
                    title="Data"
                )
                n += 1

    def draw(self, weights):

        scale = float(self.size) / np.sum(weights)
        if self.dataweights is not None:
            scale = np.sum(self.dataweights) / np.sum(weights)
        for a in self.newaxes:
            a.remove()
        self.newaxes = []

        for i in range(self.dim):
            self.axes[0, i].clear()
            newax = plot_distr1d_comparison(
                self.data[:, i],
                self.norm[:, i],
                self.bins[i],
                self.ranges[i],
                self.axes[0, i],
                self.labels[i],
                weights=scale * weights,
                dataweights=self.dataweights,
                pull=True,
                data_alpha=0.3,
                title = self.labels[i]
            )
            self.newaxes += newax

        n = 0
        for i in range(self.dim):
            for j in range(i):
                if self.dim % 2 == 0:
                    ax2 = self.axes[
                        (n // (self.dim // 2)) + 1, 2 * (n % (self.dim // 2)) + 1
                    ]
                else:
                    ax2 = self.axes[2 * (n // self.dim) + 2, n % self.dim]
                ax2.clear()
                plot_distr2d(
                    self.norm[:, i],
                    self.norm[:, j],
                    bins=(self.bins[i], self.bins[j]),
                    ranges=(self.ranges[i], self.ranges[j], self.zrange[(i, j)]),
                    fig=self.fig,
                    ax=ax2,
                    labels=(self.labels[i], self.labels[j]),
                    cmap=self.cmap,
                    weights=scale * weights,
                    colorbar=self.first,
                    title="Fit",
                )
                n += 1

        self.first = False
