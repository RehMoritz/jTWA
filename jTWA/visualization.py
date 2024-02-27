import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def create_visuals(obs, cfg):
    """
    ``create_visuals`` is a wrapper function to call other plotting functionality.

    Args:
        * ``obs``: The observables that are to be used for visualizations.
        * ``cfg``: The dictionary that contains the settings of the current run, including the working directory.
    """
    plot_deviations(obs, cfg)
    plot_spin_obs(obs, cfg)
    plot_correlation_matrices(obs, cfg)
    animate(obs, cfg)


def plot_deviations(obs, cfg):
    """
    ``plot_deviations`` plots sanity checks, making sure that conservations of particle number and energy are fulfilled.

    Args:
        * ``obs``: The observables that are to be used for visualizations.
        * ``cfg``: The dictionary that contains the settings of the current run, including the working directory.
    """
    fig, ax = plt.subplots(figsize=(7, 3))

    ax.plot(
        obs["t"][:, 0],
        obs["energy_exp"][:, 0] - obs["energy_exp"][0, 0],
        label="Energy",
    )
    ax.plot(
        obs["t"][:, 0],
        np.mean(np.sum(obs["atom_number_realspace"], axis=(2, 3)), axis=(1))
        - np.mean(np.sum(obs["atom_number_realspace"], axis=(2, 3)), axis=(1))[0],
        label="Total Atom Number Real Space",
    )
    ax.plot(
        obs["t"][:, 0],
        np.mean(np.sum(obs["atom_number_momspace"], axis=(2, 3)), axis=(1))
        - np.mean(np.sum(obs["atom_number_momspace"], axis=(2, 3)), axis=(1))[0],
        label="Total Atom Number Mom. Space",
    )
    ax.grid()
    ax.legend()
    ax.set_ylabel("Deviations over Time")
    ax.set_xlabel(r"$\langle N \rangle c_1 t$")

    fig.tight_layout()
    fig.savefig(cfg["utilParameters"]["path"] + "deviations.pdf")


def plot_spin_obs(obs, cfg):
    """
    ``plot_spin_obs`` plots the total side- and zero-mode populations as well as the different momentum-mode populations.

    Args:
        * ``obs``: The observables that are to be used for visualizations.
        * ``cfg``: The dictionary that contains the settings of the current run, including the working directory.
    """
    fig, ax = plt.subplots(nrows=2, figsize=(7, 6), sharex=True)
    ax[0].plot(
        obs["t"][:, 0],
        np.mean(obs["atom_number_realspace"][..., [0, 2]], axis=(1, 2, 3)),
        label="Side Mode",
    )
    ax[0].plot(
        obs["t"][:, 0],
        np.mean(obs["atom_number_realspace"][..., [1]], axis=(1, 2, 3)),
        label="Zero Mode",
    )

    ax[0].grid()
    ax[0].legend()
    ax[0].set_ylabel("Mode Populations")
    ax[0].set_xlabel(r"$\langle N \rangle c_1 t$")

    ax[1].plot(
        obs["t"][:, 0],
        np.mean(np.sum(obs["atom_number_momspace"], axis=(-1,)), axis=(1,)),
        label=[
            f"Momentum Mode " + str(i)
            for i in range(obs["atom_number_momspace"].shape[2])
        ],
    )

    ax[1].grid()
    ax[1].legend()
    ax[1].set_ylabel("Momentum Space Populations")
    ax[1].set_xlabel(r"$\langle N \rangle c_1 t$")

    fig.tight_layout()
    fig.savefig(cfg["utilParameters"]["path"] + "spin_obs.pdf")


def plot_correlation_matrices(obs, cfg):
    """
    ``plot_correlation_matrices`` plots correlation matrices of the two observables in ``obs["spin_obs"]`` at nine evenly distributed points in time.
    Currently this function assumes that ``obs["spin_obs"]`` only contains the observables :math:`S_x` and :math:`Q_{yz}`.

    Args:
        * ``obs``: The observables that are to be used for visualizations.
        * ``cfg``: The dictionary that contains the settings of the current run, including the working directory.
    """
    n_times, _, n_wells, n_obs = obs["spin_obs"].shape

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 15), sharex=True, sharey=True)
    idxs = np.linspace(0, n_times, 9, dtype=int)

    for counter, (idx, ax_) in enumerate(zip(idxs, ax.ravel())):
        corr_mat = np.corrcoef(
            obs["spin_obs"][idx, ...].reshape((-1, n_wells * n_obs)).T
        )
        ax_.imshow(corr_mat, vmin=-1, vmax=1, cmap="seismic")
        ax_.text(
            0.7,
            0.92,
            rf'$\langle N \rangle c_1 t = {obs["t"][idx, 0]:.1f}$',
            fontsize=14,
            weight="bold",
            transform=ax_.transAxes,
        )
        if counter % 3 == 0:
            ax_.set_yticks(np.arange(n_wells * n_obs), obs["spin_obs_names"] * n_wells)

        if counter in [6, 7, 8]:
            ax_.set_xticks(np.arange(n_wells * n_obs), obs["spin_obs_names"] * n_wells)

    fig.tight_layout()
    fig.savefig(cfg["utilParameters"]["path"] + "corrcoefs.pdf")


def animate(obs, cfg, fps=15):
    """
    ``animate`` generates a video of the evolution of the Wigner distribution given by :math:`S_x` and :math:`Q_{yz}` over time.

    Args:
        * ``obs``: The observables that are to be used for visualizations.
        * ``cfg``: The dictionary that contains the settings of the current run, including the working directory.
    """
    fig = plt.figure(figsize=(4, 4))

    scat = plt.scatter(
        obs["spin_obs"][0, :, obs["spin_obs"].shape[3] // 2, 0],
        obs["spin_obs"][0, :, obs["spin_obs"].shape[3] // 2, 1],
        alpha=0.1,
        marker="x",
    )

    txt = plt.text(
        0.64,
        0.87,
        r"$\langle N \rangle c_1 t = 0.0$",
        fontsize=14,
        weight="bold",
        transform=fig.transFigure,
    )
    plt.xlabel(r"$S_{x}$")
    plt.ylabel(r"$Q_{yz}$")
    plt.tight_layout()
    plt.xlim([-40, 40])
    plt.ylim([-40, 40])

    def animate_func(i):
        print(f'Animation progress: {i / obs["t"].shape[0]}')
        txt.set_text(rf'$\langle N \rangle c_1 t = {obs["t"][i, 0]:.1f}$')
        scat.set_offsets(
            np.array(
                [
                    obs["spin_obs"][i, :, obs["spin_obs"].shape[3] // 2, 0],
                    obs["spin_obs"][i, :, obs["spin_obs"].shape[3] // 2, 1],
                ]
            ).T
        )

    anim = animation.FuncAnimation(
        fig,
        animate_func,
        frames=obs["t"].shape[0],
        interval=1000 / fps,
    )

    anim.save(
        cfg["utilParameters"]["path"] + "evol.mp4",
        fps=fps,
        extra_args=["-vcodec", "libx264"],
    )
