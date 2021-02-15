import utils

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_gpr_1d(gpr, xs_plt=None, with_kernel=True, with_lml=True, n_samples=0, ax=None):
    if xs_plt is None:
        xs_plt = np.linspace(0, 1, num=100)

    if ax is None:
        plt.figure()
        ax = plt.gca()

    preds, pred_stds = gpr.predict(np.reshape(xs_plt, (-1, 1)), return_std=True)

    # Plot predictions
    ax.plot(xs_plt, preds, zorder=2)

    # Plot quantile band
    ax.fill_between(xs_plt, preds-1.96*pred_stds, preds+1.96*pred_stds, alpha=0.5, zorder=1)

    # Plot data points
    if hasattr(gpr, 'X_train_'):
        ax.scatter(gpr.X_train_, gpr.y_train_, c='C0', zorder=4)

        title = 'GPR kernel: %s' % gpr.kernel_ if with_kernel else ''

        if with_lml:
            title += '\n' if with_kernel else ''
            title += 'LML: %f' % gpr.log_marginal_likelihood(gpr.kernel_.theta)
    else:
        title = ''

    if n_samples > 0:
        samples = gpr.sample_y(np.reshape(xs_plt, (-1, 1)), n_samples=n_samples)
        ax.plot(xs_plt, samples, zorder=3)

    ax.set_title(title)

    return ax


def contour_gpr_2d(gpr, x1s_plt=None, x2s_plt=None, with_kernel=True, with_lml=True, fig_axes=None):
    if x1s_plt is None:
        x1s_plt = np.linspace(0, 1, num=100)

    if x2s_plt is None:
        x2s_plt = x1s_plt

    if fig_axes is None:
        fig_axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(14, 6))

    fig, (ax_gpr, ax_uncert) = fig_axes

    xs_plt = utils.cartesian_product(x1s_plt, x2s_plt)

    preds, pred_stds = gpr.predict(xs_plt, return_std=True)

    # Contour plot predictions
    cs = ax_gpr.contourf(x1s_plt, x2s_plt, preds.reshape(len(x2s_plt), len(x1s_plt)))
    fig.colorbar(cs, ax=ax_gpr)

    # Plot data points
    ax_gpr.scatter(gpr.X_train_[:, 0], gpr.X_train_[:, 1], edgecolors='w')

    # Contour plot uncertainties
    cs = ax_uncert.contourf(x1s_plt, x2s_plt, pred_stds.reshape(len(x2s_plt), len(x1s_plt)))
    fig.colorbar(cs, ax=ax_uncert)
    ax_uncert.scatter(gpr.X_train_[:, 0], gpr.X_train_[:, 1], edgecolors='w')

    # Set title
    title = 'GPR kernel: %s' % gpr.kernel_ if with_kernel else ''

    if with_lml:
        title += '\n' if with_kernel else ''
        title += 'LML: %f' % gpr.log_marginal_likelihood(gpr.kernel_.theta)

    ax_gpr.set_title(title)
    ax_uncert.set_title('Uncertainty')

    return fig, (ax_gpr, ax_uncert)


def surf_gpr_2d(gpr, x1s_plt=None, x2s_plt=None, with_kernel=True, with_lml=True, fig_axes=None):
    if x1s_plt is None:
        x1s_plt = np.linspace(0, 1, num=100)

    if x2s_plt is None:
        x2s_plt = x1s_plt

    if fig_axes is None:
        fig = plt.figure(figsize=(14, 6))
        ax_gpr = fig.add_subplot(121, projection='3d')
        ax_uncert = fig.add_subplot(122)
        fig_axes = fig, (ax_gpr, ax_uncert)

    fig, (ax_gpr, ax_uncert) = fig_axes

    xs_plt = utils.cartesian_product(x1s_plt, x2s_plt)

    preds, pred_stds = gpr.predict(xs_plt, return_std=True)

    # Surface plot predictions
    ax_gpr.plot_surface(*np.meshgrid(x1s_plt, x2s_plt),
                        preds.reshape(len(x2s_plt), len(x1s_plt)),
                        alpha=0.25)

    # Plot data points
    ax_gpr.scatter(gpr.X_train_[:, 0], gpr.X_train_[:, 1], gpr.y_train_)

    # Contour plot uncertainties
    cs = ax_uncert.contourf(x1s_plt, x2s_plt, pred_stds.reshape(len(x2s_plt), len(x1s_plt)))
    fig.colorbar(cs, ax=ax_uncert)
    ax_uncert.scatter(gpr.X_train_[:, 0], gpr.X_train_[:, 1], edgecolors='w')

    # Set title
    title = 'GPR kernel: %s' % gpr.kernel_ if with_kernel else ''

    if with_lml:
        title += '\n' if with_kernel else ''
        title += 'LML: %f' % gpr.log_marginal_likelihood(gpr.kernel_.theta)

    ax_gpr.set_title(title)
    ax_uncert.set_title('Uncertainty')

    return fig, (ax_gpr, ax_uncert)
