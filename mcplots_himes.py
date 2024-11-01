#!/opt/anaconda3/bin/python

import sys, os, platform
import six
import numpy as np
import scipy.stats as stats
import scipy.interpolate as si
import matplotlib as mpl
# mpl.use("TkAgg")
import matplotlib.pyplot as plt
import credible_region as cr


def credregion(posterior, percentile=[0.68269, 0.95450, 0.99730], 
               lims=(None,None), numpts=100):
    """
    Computes the credible region of a 1D posterior for given percentiles.
    Inputs
    ------
    posterior : 1d array of parameter value at each iteration.
    percentile: 1D float ndarray, list, or float.
                The percentile (actually the fraction) of the credible region.
                A value in the range: (0, 1).
    lims: tuple, floats. Minimum and maximum allowed values for posterior. 
                         Should only be used if there are physically-imposed 
                         limits.
    numpts: int. Number of points to use when calculating the PDF.
                 Note: large values have significant compute cost.
    Outputs
    -------
    pdf : array. Probability density function.
    xpdf: array. X values associated with `pdf`.
    CRlo: list.  Lower bounds on credible region(s).
    CRhi: list.  Upper bounds on credible region(s).
    """
    # Make sure `percentile` is a list or array
    if type(percentile) == float:
        percentile = np.array([percentile])

    # Compute the posterior's PDF:
    kernel = stats.gaussian_kde(posterior)
    # Use a Gaussian kernel density estimate to trace the PDF:
    # Interpolate-resample over finer grid (because kernel.evaluate
    #  is expensive):
    if lims[0] is not None:
        lo = min(np.amin(posterior), lims[0])
    else:
        lo = np.amin(posterior)
    if lims[1] is not None:
        hi = max(np.amax(posterior), lims[1])
    else:
        hi = np.amax(posterior)
    x    = np.linspace(lo, hi, numpts)
    f    = si.interp1d(x, kernel.evaluate(x))
    xpdf = np.linspace(lo, hi, 100*numpts)
    pdf  = f(xpdf)


    # Sort the PDF in descending order:
    ip = np.argsort(pdf)[::-1]
    # Sorted CDF:
    cdf = np.cumsum(pdf[ip])

    # List to hold boundaries of CRs
    # List is used because a given CR may be multiple disconnected regions
    CRlo = []
    CRhi = []
    # Find boundary for each specified percentile
    for i in range(len(percentile)):
        # Indices of the highest posterior density:
        iHPD = np.where(cdf >= percentile[i]*cdf[-1])[0][0]
        # Minimum density in the HPD region:
        HPDmin   = np.amin(pdf[ip][0:iHPD])
        # Find the contiguous areas of the PDF greater than or equal to HPDmin
        HPDbool  = pdf >= HPDmin
        idiff    = np.diff(HPDbool) # True where HPDbool changes T to F or F to T
        iregion, = idiff.nonzero()  # Indexes of Trues. Note , because returns tuple
        # Check boundaries
        if HPDbool[0]:
            iregion = np.insert(iregion, 0, -1) # This -1 is changed to 0 below when 
        if HPDbool[-1]:                       #   correcting start index for regions
            iregion = np.append(iregion, len(HPDbool)-1)
        # Reshape into 2 columns of start/end indices
        iregion.shape = (-1, 2)
        # Add 1 to start of each region due to np.diff() functionality
        iregion[:,0] += 1
        # Store the min and max of each (possibly disconnected) region
        CRlo.append(xpdf[iregion[:,0]])
        CRhi.append(xpdf[iregion[:,1]])

    return pdf, xpdf, CRlo, CRhi



def pairwise(allparams, title=None, parname=None, thinning=1,
             fignum=-11, savefile=None, style="hist", fs=34, nbins=20, 
             truepars=None, credreg=False, ptitle=False, ndec=None):
  """
  Plot parameter pairwise posterior distributions

  Parameters
  ----------
  allparams: 2D ndarray
     An MCMC sampling array with dimension (number of parameters,
     sampling length).
  title: String
     Plot title.
  parname: Iterable (strings)
     List of label names for parameters.  If None use ['P0', 'P1', ...].
  thinning: Integer
     Thinning factor for plotting (plot every thinning-th value).
  fignum: Integer
     The figure number.
  savefile: Boolean
     If not None, name of file to save the plot.
  style: String
     Choose between 'hist' to plot as histogram, or 'points' to plot
     the individual points.
  fs: Int
     Font size
  nbins: Int
     Number of bins for 2D histograms. 1D histograms will use 2*bins
  truepars: array.
     True parameters, if known.  Plots them in red.
  credreg: Boolean
     Determines whether or not to plot the credible regions.
  ptitle: Boolean.
     Controls subplot titles.
     If False, will not plot title. 
     If True, will plot the title as the 50% quantile +- the 68.27% region.
     If the 68.27% region consists of multiple disconnected regions, the 
     greatest extent will be used for the title.
     If `truepars` is given, the title also includes the true parameter value.
  ndec: None, or array of ints.
     If None, does nothing.
     If an array of ints, sets the number of decimal places to round each value
     in the title.

  Uncredited developers
  ---------------------
  Kevin Stevenson  (UCF)
  Ryan Hardy  (UCF)
  """
  # Get number of parameters and length of chain:
  npars, niter = np.shape(allparams)

  # Ensure proper number of parameters for `truepars`
  if truepars is not None:
    if len(truepars) != npars:
      raise ValueError("True parameters passed to pairwise().\n" +           \
                       "But, it does not match the posterior dimensionality.")

  # Don't plot if there are no pairs:
  if npars == 1:
    return

  # Set default parameter names:
  if parname is None:
    namelen = int(2+np.log10(np.amax([npars-1,1])))
    parname = np.zeros(npars, "|S%d"%namelen if six.PY2 else "<U%d"%namelen)
    for i in np.arange(npars):
      parname[i] = "P" + str(i).zfill(namelen-1)

  # Set palette color:
  palette = mpl.cm.get_cmap('RdPu', 256)
  palette.set_under(alpha=0.0)
  palette.set_bad(alpha=0.0)

  if credreg:
    hkw = {'histtype':'step', 'lw':1.0, 'color':'#5c026e'}
    cr_alpha = [1.0, 0.65, 0.4]
  else:
    hkw = {'edgecolor':'purple', 'color':'#5c026e'}

  fig = plt.figure(fignum, figsize=(15, 15))
  plt.clf()
  if title is not None:
    plt.suptitle(title, size=fs+4)

  h = 1 # Subplot index
  plt.subplots_adjust(left=0.15,   right=0.95, bottom=0.15, top=0.9,
                      hspace=0.20, wspace=0.20)

  for   j in np.arange(npars): # Rows
    for i in np.arange(npars): # Columns
      if j > i or j == i:
        a = plt.subplot(npars, npars, h)
        # Y labels:
        if not i:
            plt.ylabel(parname[j], size=fs, rotation=0, horizontalalignment='right', verticalalignment='center')
            if j:
              plt.yticks(size=fs-8)
            else:
              plt.yticks(visible=False)
        else:
          a = plt.yticks(visible=False)
        # X labels:
        if j == npars-1:
          plt.xticks(size=fs-8, rotation=90)
          plt.xlabel(parname[i], size=fs, rotation=30, horizontalalignment='right', verticalalignment='top')
        else:
          a = plt.xticks(visible=False)
        # The plot:
        if style=="hist":
          # 2D histogram
          if j > i:
            hist2d, xedges, yedges = np.histogram2d(allparams[i, 0::thinning],
                                                    allparams[j, 0::thinning], 
                                                    nbins, density=False)
            vmin = 0.0
            hist2d[np.where(hist2d == 0)] = np.nan
            a = plt.imshow(hist2d.T, extent=(xedges[0], xedges[-1], yedges[0],
                           yedges[-1]), cmap=palette, vmin=vmin, aspect='auto',
                           origin='lower', interpolation='bilinear')
            if truepars is not None: # plot true params
              plt.plot(truepars[i], truepars[j], '*', color='black', ms=20, 
                       markeredgecolor='white', markeredgewidth=1)
          # 1D histogram
          else:
            vals, bins, hh = plt.hist(allparams[i,0::thinning], 2*nbins, 
                                     density=False, **hkw)
        elif style=="points":
          if j > i:
            a = plt.plot(allparams[i], allparams[j], ",")
            if truepars is not None: # plot true params
              plt.plot(truepars[i], truepars[j], '*', color='black', ms=20, 
                       markeredgecolor='black', markeredgewidth=1)
          else:
            vals, bins, hh = plt.hist(allparams[i,0::thinning], 2*nbins, 
                                     density=False, **hkw)
        # Plotting credible regions, true params for 1D hist, and titles
        if j <= i:
          if credreg:
            pdf, xpdf, CRlo, CRhi = cr.credregion(allparams[i,0::thinning], 
                                         percentile=[0.68269]) #percentile=[0.68269, 0.95450, 0.99730]
            vals = np.r_[0, vals, 0]
            bins = np.r_[bins[0] - (bins[1]-bins[0]), bins]
            f    = si.interp1d(bins+0.5 * (bins[1]-bins[0]), vals, kind='nearest')
            # Plot credible regions as shaded areas
            for k in range(len(CRlo)):
              for r in range(len(CRlo[k])):
                plt.gca().fill_between(xpdf, 0, f(xpdf),
                                       where=(xpdf>=CRlo[k][r]) * \
                                             (xpdf<=CRhi[k][r]), 
                                       facecolor=(0.9725490196078431, 0.6549019607843137, 0.7137254901960784, cr_alpha[k]), 
                                       edgecolor='none', interpolate=False, 
                                       zorder=-2+2*k)
            if ptitle:
              med = np.median(allparams[i,0::thinning])
              if ndec is not None:
                lo  = np.around(CRlo[0][ 0]-med, ndec[i])
                hi  = np.around(CRhi[0][-1]-med, ndec[i])
                med = np.around(med, ndec[i])
              else:
                lo  = CRlo[0][ 0]-med
                hi  = CRhi[0][-1]-med
              titlestr = parname[i] + r' = $'+str(med)+'_{{{0:+g}}}^{{{1:+g}}}$'.format(lo, hi)
          if truepars is not None:
            plt.gca().axvline(truepars[i], color='black', lw=3) # plot true param
            if ptitle and credreg:
              if ndec is not None:
                titlestr = 'True value: '+str(np.around(truepars[i], ndec[i]))+\
                           '\n'+titlestr
              else:
                titlestr = 'True value: '+str(truepars[i])+'\n'+titlestr
          if ptitle and credreg:
            plt.title(titlestr, fontsize=fs-18)
          if i==0:
            if credreg:
              # Add labels & legend
              sig1 = mpl.patches.Patch(color=(0.9725490196078431, 0.6549019607843137, 0.7137254901960784, cr_alpha[0]), 
                                       label='$68.27\%$ region')
              # sig2 = mpl.patches.Patch(color=(0.9725490196078431, 0.6549019607843137, 0.7137254901960784, cr_alpha[1]), 
              #                          label='$95.45\%$ region')
              # sig3 = mpl.patches.Patch(color=(0.9725490196078431, 0.6549019607843137, 0.7137254901960784, cr_alpha[2]), 
              #                          label='$99.73\%$ region')
              # hndls = [sig1, sig2, sig3]
              hndls = [sig1]
            else:
              hndls = []
            if truepars is not None:
              hndls = hndls + [mpl.lines.Line2D([], [], color='black', lw=4, 
                                                marker='*', ms=20, 
                                                markeredgecolor='black', 
                                                markeredgewidth=1, 
                                                label='True value')]
            if credreg or truepars is not None:
              plt.legend(handles=hndls, prop={'size':fs-14}, 
                         bbox_to_anchor=(1, 1.05), ncol=2)

        plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
        plt.gca().yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))

      h += 1

  # Align labels
  fig.align_labels()

  # The colorbar:
  if style == "hist":
    if npars > 2:
      a = plt.subplot(2, 6, 5, frameon=False)
      a.yaxis.set_visible(False)
      a.xaxis.set_visible(False)
    bounds = np.linspace(0, 1.0, 64)
    norm = mpl.colors.BoundaryNorm(bounds, palette.N)
    ax2 = fig.add_axes([0.8, 0.5, 0.025, 0.36])
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=palette, norm=norm,
          spacing='proportional', boundaries=bounds, format='%.1f')
    cb.set_label("Normalized point density", fontsize=fs)
    cb.set_ticks(np.linspace(0, 1, 5))
    cb.ax.tick_params(labelsize=fs-8)
    plt.draw()

  # Save file:
  if savefile is not None:
    plt.savefig(savefile, bbox_inches='tight')
    plt.close()

# 5 vs. 6 parameters being fit
NFIT = 5

NSAMP = 10000
test = np.zeros((6, NSAMP), dtype=float)
test[0] = np.random.normal(0.5, 0.1, NSAMP)
test[1] = np.random.normal( -3, 0.5, NSAMP)
test[2] = np.random.normal( -5, 1  , NSAMP)
test[3] = np.random.normal( -5, 1  , NSAMP)
test[4] = np.random.normal(  1, 0.1, NSAMP)
test[5] = np.random.normal(0, 0.2, NSAMP)

true = np.array([0.5, -3, -5, -5, 1, 0])

if NFIT == 5:
    test = test[1:]
    true = true[1:]

pairwise(test, savefile='test', truepars=true, credreg=True)


