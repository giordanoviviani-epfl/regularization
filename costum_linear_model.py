import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
from scipy.special import legendre
import ipywidgets as widgets


class PolyModel():

    def __init__(self):
        pass
 

    def design_matrix(self, x, ncol):
        return np.hstack([legendre(n)(np.array(x)).reshape(-1, 1) for n in range(ncol)])
    

    def interactive(self, x, y, regularize=None, max_order=20, over_plot=None, _step=1):

        if regularize is None:
            log_lam = [0]
        elif str(regularize).lower() == 'l1' or str(regularize).lower() == 'l2':
            log_lam = np.arange(-18, 3, 0.5)
        else:
            raise ValueError(f"Invalid regularizer: {regularize}")

        assert len(x) == len(y), 'x and y have not the same dimensions'

        if over_plot is not None:
            x_test, y_test = over_plot
            plot_test_set = True

        slider = widgets.IntSlider(
                    value=0,
                    min=0,
                    max=max_order,
                    step=_step,
                    disabled=False,
                    continuous_update=False,
                    orientation='horizontal',
                    readout=False,
                    layout=widgets.Layout(width='90%'),
                    description='Number of orders:'
                    )

        s_text = widgets.Label(value="{:.2f}".format(0))

        l_slider = widgets.IntSlider(
                    value=0,
                    min=0,
                    max=len(log_lam) - 1,
                    step=1,
                    disabled=False,
                    continuous_update=False,
                    orientation="horizontal",
                    readout=False,
                    layout=widgets.Layout(width="90%"),
                    description="log lambda"
                    )

        l_text = widgets.Label(value="{:.2f}".format(log_lam[0]))

        def visualize_func(N=0, log_lam_idx=0):

            s_text.value = "{:d}".format(N)
            l_text.value = "{:.2f}".format(log_lam[log_lam_idx])

            # Compute the weights
            A_train = self.design_matrix(x, N + 1)
            if str(regularize).lower() == "none":
                w = lstsq(A_train, y)
            elif str(regularize).lower() == "l1":
                w = L1(A_train, y, log_lam[log_lam_idx])
            elif str(regularize).lower() == "l2":
                w = L2(A_train, y, log_lam[log_lam_idx])
            else:
                raise ValueError(f"Invalid regularizer: {regularize}")
            model_train = A_train.dot(w)

            # Compute the prediction
            A_test = self.design_matrix(x_test, N + 1)
            model_test = A_test.dot(w)

            # Compute the model on a high res grid
            x_hires = np.linspace(
                np.concatenate((x, x_test)).min(),
                np.concatenate((x, x_test)).max(),
                300,
            )
            A_hires = self.design_matrix(x_hires, N + 1)
            model_hires = A_hires.dot(w)

            # Set up the plot
            fig = plt.figure(figsize=(15, 8))
            fig.subplots_adjust(wspace=0.25)
            ax = fig.subplot_mosaic(
                    """
                    AAB
                    AAC
                    """)

            ax["A"].set_xlabel("x", fontsize=28)
            ax["A"].set_ylabel("y(x)", fontsize=28)
            ax["A"].set_xlim(-0.5, 0.5)
            ymin = np.min(y)
            ymax = np.max(y)
            if plot_test_set:
                ymin = min((ymin, np.min(y_test)))
                ymax = max((ymax, np.max(y_test)))
            ypad = 0.5 * (ymax - ymin)
            ax["A"].set_ylim(ymin - ypad, ymax + ypad)

            # Plot the data
            ax["A"].plot(x, y, "ko")
            if plot_test_set:
                ax["A"].plot(x_test, y_test, "C1o")

            # Plot the model
            x = np.concatenate((x, x_test, x_hires))
            m = np.concatenate((model_train, model_test, model_hires))
            idx = np.argsort(x)
            x = x[idx]
            m = m[idx]
            ax["A"].plot(x, m, "C0-")

            # Print the loss
            loss_train = np.sum((y - model_train) ** 2) / len(y)
            ax["B"].text( 0.1, 0.5, f"Train loss: {loss_train:.2e}", ha="left", fontsize=20)

            if plot_test_set:
                loss_test = np.sum((y_test - model_test) ** 2) / len(y_test)
                ax["B"].text(0.1, 0.35, f"Test loss:  {loss_test:.2e}", ha="left", fontsize=20)
            ax["B"].axis("off")

            # Plot the weights
            ax["C"].plot(np.log10(np.abs(w)), "C1-")
            ax["C"].plot(np.log10(np.abs(w)), "k.")
            ax["C"].axhline(0, color="k", lw=1, alpha=0.5, ls="--")
            ax["C"].set_xlim(0, max_order)
            ax["C"].set_ylim(-15, 15)
            ax["C"].set_ylabel("log abs weights", fontsize=16)
            ax["C"].set_xlabel("weight index", fontsize=16)

        plot = widgets.interactive_output( visualize_func, {"N": slider, "log_lam_idx": l_slider})

        # Display!
        display(plot)
        display(widgets.HBox([slider, s_text]))
        if str(regularize).lower() != "none":
            display(widgets.HBox([l_slider, l_text]))

class FSModel(PolyModel):

    def design_matrix(self, x, ncol):
        columns = []
        for n in range(ncol):
            if n == 0:
                columns.append(np.ones(len(x)).reshape(-1, 1))
            elif n%2 == 1:
                columns.append(np.cos(2.*np.pi*x*(n//2 + 1)))
            elif n%2 == 0:
                columns.append(np.sin(2.*np.pi*x*(n//2)))
            else:
                ValueError(f"Invalid value for n: {n}")

        return np.hstack(columns)

    def interactive(self, x, y, regularize=None, max_order=20, over_plot=None, _step=2):
        return super().interactive(x, y, regularize, max_order, over_plot, _step)


def L1(A, y, log_lam=5.0, maxiter=9999, eps=1e-15, tol=1e-8):
    """L1 regularized least squares via iterated ridge (L2) regression.

    See Section 2.5 of

        https://www.cs.ubc.ca/~schmidtm/Documents/2005_Notes_Lasso.pdf

    The basic idea is to iteratively zero out the prior on the weights
    until convergence.

    Args:
        A (ndarray): The design matrix for the linear problem, shape ``(N, M)``.
        y (ndarray): The data vector, length ``N``.
        log_lam (float or ndarray, optional): The log of the regularization
            strength parameter, ``lambda``. This may either be a scalar or
            a vector of length ``N``. Defaults to 5.0.
        maxiter (int, optional): Maximum number of iterations. Defaults to 9999.
        eps (float, optional): Precision of the algorithm. Defaults to 1e-15.
        tol (float, optional): Iteration stop tolerance. Defaults to 1e-8.

    Returns:
        ndarray: The vector of coefficients ``w`` that minimizes the L1 norm
            for the linear problem.
    """
    ATA = A.T @ A
    ATy = A.T @ y
    w = np.ones_like(ATA[0])
    lam = 10 ** log_lam
    for _ in range(maxiter):
        absw = np.abs(w)
        if hasattr(lam, "__len__"):
            absw[absw < lam * eps] = lam[absw < lam * eps] * eps
        else:
            absw[absw < lam * eps] = lam * eps
        KInv = np.array(ATA)
        KInv[np.diag_indices_from(KInv)] += lam / absw
        try:
            w_new = np.linalg.solve(KInv, ATy)
        except np.linalg.LinAlgError:
            w_new = np.linalg.lstsq(KInv, ATy, rcond=None)[0]
        chisq = np.sum((w - w_new) ** 2)
        w = w_new
        if chisq < tol:
            break
    w[np.abs(w) < tol] = 1e-15
    return w


def L2(A, y, log_lam=5.0):
    """L2 regularized least squares solver.

    Args:
        A (ndarray): The design matrix for the linear problem, shape ``(N, M)``.
        y (ndarray): The data vector, length ``N``.
        log_lam (float or ndarray, optional): The log of the regularization
            strength parameter, ``lambda``. This may either be a scalar or
            a vector of length ``N``. Defaults to 5.0.

    Returns:
        ndarray: The vector of coefficients ``w`` that minimizes the L2 norm
            for the linear problem.
    """
    ATA = A.T @ A
    ATy = A.T @ y
    ATA[np.diag_indices_from(ATA)] += 10 ** log_lam
    try:
        w = np.linalg.solve(ATA, ATy)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(ATA, ATy, rcond=None)[0]
    return w


def lstsq(A, y):
    """Unregularized least squares solver.

    Args:
        A (ndarray): The design matrix for the linear problem, shape ``(N, M)``.
        y (ndarray): The data vector, length ``N``.

    Returns:
        ndarray: The vector of coefficients ``w`` that minimizes the chi squared
            loss.
    """
    ATA = A.T @ A
    ATy = A.T @ y
    try:
        w = np.linalg.solve(ATA, ATy)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(ATA, ATy, rcond=None)[0]
    return w
