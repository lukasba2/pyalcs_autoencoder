import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_exploit_results(df):
    expl_df = df[df['phase'] == 'exploit']

    fig, axs = plt.subplots(3, 3, figsize=(18, 16))

    # or expl_df.index.get_level_values(0).unique()
    for algno, alg in enumerate(['ACS2', 'AACS2_v1', 'AACS2_v2']):
        alg_df = expl_df.loc[alg]

        idx = pd.Index(name='exploit trial',
                       data=np.arange(1, len(alg_df) + 1))
        alg_df.set_index(idx, inplace=True)

        axs[0, algno].set_title(f'Steps ({alg})')
        alg_df['steps_in_trial'].plot(ax=axs[0, algno])

        axs[1, algno].set_title(f'Population ({alg})')
        alg_df['population'].plot(ax=axs[1, algno])
        alg_df['reliable'].plot(ax=axs[1, algno])
        axs[1, algno].legend()

        axs[2, algno].set_title(f'Rho ({alg})')
        alg_df['rho'].plot(ax=axs[2, algno])

    plt.tight_layout(h_pad=3.0)




