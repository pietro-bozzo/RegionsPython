''' Plotting utilities for publication grade figures ''' 

import matplotlib.pyplot


def adjustAxes(axs):
    # adjust axes properties to emprove figure appearance
    #
    # arguments:
    #     axs    collection of matplotlib axes

    if isinstance(axs,matplotlib.axes._axes.Axes):
        axs = [axs]

    for ax in axs:
        
        # remove upper and right borders
        ax.spines[['right', 'top']].set_visible(False)

        # adjust thickness
        for spine in ['top','bottom','left','right']:
            ax.spines[spine].set_linewidth(1.3)
        ax.tick_params(width=1.3)

    return