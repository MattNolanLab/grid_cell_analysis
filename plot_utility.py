import matplotlib.pylab as plt
import math
import numpy as np
import random
import PostSorting.parameters
prm = PostSorting.parameters.Parameters()


'''
colour functions are from https://gist.github.com/adewes/5884820
'''


def style_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    return plt, ax


def style_open_field_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        right=False,
        left=False,
        labelleft=False,
        labelbottom=False)  # labels along the bottom edge are off

    ax.set_aspect('equal')
    return ax


def style_polar_plot(ax):
    ax.spines['polar'].set_visible(False)
    ax.set_yticklabels([])  # remove yticklabels
    # ax.grid(None)
    plt.xticks([math.radians(0), math.radians(90), math.radians(180), math.radians(270)])
    ax.axvline(math.radians(90), color='black', linewidth=1, alpha=0.6)
    ax.axvline(math.radians(180), color='black', linewidth=1, alpha=0.6)
    ax.axvline(math.radians(270), color='black', linewidth=1, alpha=0.6)
    ax.axvline(math.radians(0), color='black', linewidth=1, alpha=0.6)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi/2.0)
    ax.xaxis.set_tick_params(labelsize=25)
    return ax


def get_random_color(pastel_factor = 0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]


def color_distance(c1,c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])


def generate_new_color(existing_colors, pastel_factor=0.5):
    max_distance = None
    best_color = None
    for i in range(0, 100):
        color = get_random_color(pastel_factor = pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color, c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color



def adjust_spine_thickness(ax):
    for axis in ['left','bottom']:
        ax.spines[axis].set_linewidth(1)


def adjust_spines(ax,spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward',0)) # outward by 10 points
            #spine.set_smart_bounds(True)
        else:
            spine.set_color('none') # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])


def get_weights_normalized_hist(array_in):
    weights = np.ones_like(array_in) / float(len(array_in))
    return weights


def format_bar_chart(ax, x_label, y_label):
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.gcf().subplots_adjust(left=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel(x_label, fontsize=25)
    ax.set_ylabel(y_label, fontsize=25)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    return ax


def plot_cumulative_histogram(corr_values, ax, color='black', number_of_bins=40):
    plt.xlim(-1, 1)
    plt.yticks([0, 1])
    ax = format_bar_chart(ax, 'r', 'Cumulative probability')
    values, base = np.histogram(corr_values, bins=number_of_bins, range=(-1, 1))
    # evaluate the cumulative
    cumulative = np.cumsum(values / len(corr_values))
    # plot the cumulative function
    plt.plot(base[:-1], cumulative, c=color, linewidth=5, alpha=0.6)
    return ax


def plot_cumulative_histogram_from_zero(corr_values, ax, color='black', number_of_bins=40):
    plt.xlim(0, 1)
    plt.yticks([0, 1], fontsize=20)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.gcf().subplots_adjust(left=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.xlabel('Percentile score', fontsize=25)
    plt.ylabel('Cumulative probability', fontsize=25)
    # ax.xaxis.set_tick_params(labelsize=20)
    # ax.yaxis.set_tick_params(labelsize=20)
    plt.xticks([0, 1], ["0", "100"], fontsize=20)
    values, base = np.histogram(corr_values, bins=number_of_bins, range=(-1, 1))
    # evaluate the cumulative
    cumulative = np.cumsum(values / len(corr_values))
    # plot the cumulative function
    plt.plot(base[:-1], cumulative, c=color, linewidth=5, alpha=0.6)
    return ax
