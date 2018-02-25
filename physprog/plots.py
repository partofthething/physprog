"""Collection of plot functions."""

import matplotlib.pyplot as plt
import numpy as np

# pylint: disable=too-many-locals
def plot_optimization_results(preferences, initials, finals):
    """Show how each design parameter changed in terms of preference."""
    labels = [
        'Highly\nUndesirable', 'Undesirable', 'Tolerable', 'Desirable',
        'Highly\ndesirable'
    ]
    _fig, ax = plt.subplots()
    xticks = []
    y_vals = np.arange(len(preferences)) + 0.5
    x_max = max(y_vals)  # to keep squareish
    for i, (initial, final, (_prefname, pref)) in enumerate(
            zip(initials, finals, preferences.items())):
        y = y_vals[len(preferences) - i - 1]
        initial_f = pref.acceptability(initial) * x_max
        final_f = pref.acceptability(final) * x_max
        # ax.text(-0.3, y, prefname)
        ax.annotate(
            '',
            xy=(final_f, y),
            xytext=(initial_f, y),
            arrowprops={'arrowstyle': '->'})
        circle1 = plt.Circle((initial_f, y), 0.1, color='k', fill=False)
        circle2 = plt.Circle((final_f, y), 0.1, color='k', fill=True)
        ax.add_artist(circle1)
        ax.add_artist(circle2)
        # plt.arrow(initial_f, y, (final_f - initial_f), 0.0, width=0.01)
    for i, _label in enumerate(labels):
        xtick = (i + 1.0) * x_max / 5.0
        ax.axvline(xtick, color='k', alpha=0.2, linestyle='--')
        xticks.append((i + 0.5) * x_max / 5.0)  # shift to middle
    # ax.axvspan(0, 0.2, alpha=0.2, color='green')
    ax.set_xlim([0, x_max + 1])
    ax.set_ylim([0, x_max + 0.5])

    plt.xticks(xticks, labels)
    plt.yticks(
        list(reversed(y_vals)), [s.capitalize() for s in preferences.keys()])
    plt.title('Optimization Results')
    plt.tight_layout()
    plt.show()
