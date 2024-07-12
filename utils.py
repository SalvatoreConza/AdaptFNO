from typing import List, Tuple
import matplotlib.pyplot as plt
import torch


def plot_2d(
    *states: Tuple[torch.Tensor], 
    timesteps: List[int],
    dim_names: List[str],
    filename: str,
):

    for state in states:
        assert state.ndim == 3
        assert len(timesteps) == len(states)
        state.to(device='cpu')

    u_dim: int = state.shape[0]
    x_dim: int = state.shape[1]
    y_dim: int = state.shape[2]

    assert len(dim_names) == u_dim

    fig, axs = plt.subplots(len(timesteps), u_dim, figsize=(5 * u_dim, 5 * len(timesteps)))

    for t_idx, t in enumerate(timesteps):
        for dim, dim_name in enumerate(dim_names):
            axs[t_idx, dim].imshow(
                states[t_idx][dim],
                aspect="auto",
                origin="lower",
                extent=[-1., 1., -1., 1.],
            )
            axs[t_idx, dim].set_xticks([])
            axs[t_idx, dim].set_yticks([])
            axs[t_idx, dim].set_title(f"${dim_name}(t={t})$")

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.01, top=0.99, wspace=0.2, hspace=0.2)
    plt.savefig(filename)



if __name__ == '__main__':
    from datasets import OneShotDiffReact2d
    dataset = OneShotDiffReact2d(
        dataroot='data/2D/diffusion-reaction/2D_diff-react_NA_NA.h5'
    )

    timesteps = [[0, 5], [10, 20], [30, 40], [50, 60], [70, 80], [90, 100]]

    tensors = []
    for start_step, end_step in timesteps:
        dataset = OneShotDiffReact2d(
            dataroot='data/2D/diffusion-reaction/2D_diff-react_NA_NA.h5',
            input_step=start_step,
            target_step=end_step,
        )
        index = 0
        input, target = dataset[index]
        tensors.extend([input, target])

    plot_2d(
        *tensors,
        timesteps=[0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        dim_names=['u_1', 'u_2'],
        filename='sample.png'
    )


    # plot_2d(
    #     input=input, 
    #     target=target, 
    #     dim_names=['u_1', 'u_2'],
    #     timesteps=[5, 10],
    #     filename='test.png',
    # )


