import numpy as np
from pytorch_rl_collection.utils import fetch_uniform_unscented_transfo
from sklearn.manifold import TSNE
import matplotlib
#matplotlib.use("TkAgg")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIMS = [2, 12, 21, 22, 31]
n_rows, n_columns = 3, 2
WIDTH_SIZE, HEIGHT_SIZE = 12, 9
fig = plt.figure(figsize=(WIDTH_SIZE, HEIGHT_SIZE))
gs = fig.add_gridspec(n_rows, n_columns, hspace=0.3)
axs = np.array([[None]*n_columns]*n_rows)

colors = ['b', 'r']#, 'g', 'y']
transparencies = [1., 1.]
markers = ["o", "x"]
labels = ["Sigma-points for Training", "Sigma-points for Evaluations"]

i, j = 0, 0
for n, dim in enumerate(DIMS):
    print(dim, (i, j))
    axs[i, j] = fig.add_subplot(gs[i, j])
    X = fetch_uniform_unscented_transfo(dim=dim)["sigma_points"][:2]; X.shape
    if dim == 2:
        print(X)
    #####
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', random_state=0, perplexity=3)
    for idx, elt in enumerate(X):
        if dim > 2:
          Y = tsne.fit_transform(elt)
        else:
          Y = elt
        Y.shape
        axs[i, j].scatter(Y[:, 0], Y[:, 1], c=colors[idx], alpha=transparencies[idx], marker=markers[idx], label=labels[idx])
    #####
    axs[i, j].set_title("Dim = {}{}".format(dim, " - t-SNE projection" if dim > 2 else ""))
    #axs[i, j].legend()
    #####
    j = (j + 1) % n_columns
    if j == 0:
        i = (i + 1) % n_rows

#plt.legend()
axs[i, j] = fig.add_subplot(gs[i, j])
# Create a color palette
palette = dict(zip(labels, colors))
# Create legend handles manually
handles = [matplotlib.patches.Patch(color=palette[x], label=x) for x in palette.keys()]
# Create legend
axs[i, j].legend(handles=handles)
# Get current axes object and turn off axis
axs[i, j].set_axis_off()

#plt.show()
plt.tight_layout()
plt.savefig("./viz_ut_sigmapoints.pdf")
