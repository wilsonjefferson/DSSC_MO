import matplotlib.pyplot as plt
import numpy as np


def plot_scalability(results:list, save_at:str) -> None:

    results_array = np.array(results)
    n_fields_instances = np.unique(results_array[:, 1])

    plt.figure(figsize=(10, 6))

    for value in n_fields_instances:
        # filter rows with the current unique value in the storages column
        subset = results_array[results_array[:, 1] == value]
        plt.plot(subset[:, 0], subset[:, 2], label=f'num. fields: {int(value)}', marker='o')

    plt.title('Runtime trend for each fix number of fields')
    plt.xlabel('Number of storages')
    plt.ylabel('Runtime (sec.)')
    plt.legend()

    plt.savefig(save_at, format='svg', bbox_inches='tight')
    plt.show()