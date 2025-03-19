import matplotlib.pyplot as plt
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
from collections import defaultdict

import pandas as pd

# Parameters
n_dimensions = 2  # For 2D
p = 4  # Order of the Hilbert curve (2^p bins per dimension for a total of 2^(p*n_dimensions))
bin_size = 1/(2**p)  # Size of each bin
hilbert_curve = HilbertCurve(p, n_dimensions)

# Generate some sample points
np.random.seed(0)
a, b = 0., 1.
samples = a + (b - a) * np.random.rand(10000, 2)  # 1000 random points in 2D space

# Initialize counters
counters = defaultdict(int)

# Map samples to Hilbert indices
def get_hilbert_indices(hilbert_curve, samples, bin_size):
    hilbert_indices = []
    for sample in samples:
        scaled_sample = [int(coord // bin_size) for coord in sample]
        hilbert_index = hilbert_curve.distance_from_point(scaled_sample)
        hilbert_indices.append(hilbert_index)
    return hilbert_indices

# Update counters for the samples
def update_counters(hilbert_curve, samples, bin_size, counters):
    hilbert_indices = get_hilbert_indices(hilbert_curve, samples, bin_size)
    for index in hilbert_indices:
        counters[index] += 1

#
algo = "c"
rnd_seed_id = 4
if algo == "c":
    date = "20241127"
    #date = "20241202"
    path = "./trained_agents/cmdsac_UT_EX_her/Walker2d-v3_body_mass_dof_armature_{}/{}/checkpoint/minibatch_params_count.csv".format(
        date, rnd_seed_id
    )
elif algo == "e":
    date = "20241125"
    path = "./trained_agents/emdsac_UT_EX_her/Walker2d-v3_body_mass_dof_armature_{}/{}/checkpoint/minibatch_params_count.csv".format(
        date, rnd_seed_id
    )

counters_data = pd.read_csv(
    path,
    index_col=0
)
print(counters_data)
counters_data = counters_data.to_dict()
print(len(counters_data.keys()))

total_count = 0
max_count = -np.inf
min_count = np.inf
for hilbert_index, count in counters_data.items():
    #print(hilbert_index, count, count[0])
    #print(type(hilbert_index), type(count))
    max_count = max(max_count, count[0])
    min_count = min(min_count, count[0])
    total_count += count[0]

print(total_count)

#"""
for hilbert_index, count in counters_data.items():
    counters[hilbert_index] = (count[0]) / (max_count) #count[0] / total_count

#"""

#"""
# Prepare data for visualization
max_index = 2**(p * n_dimensions)  # Total number of regions
print(max_index)
heatmap = np.zeros((2**p, 2**p))

for hilbert_index, count in counters.items():
    # Convert Hilbert index back to 2D coordinates
    coord = hilbert_curve.point_from_distance(hilbert_index)
    heatmap[coord[0], coord[1]] = count

# Plot heatmap
plt.figure(figsize=(10, 8))
plt.imshow(heatmap, origin='lower', cmap='viridis')
plt.colorbar(label="Counts")
plt.title("Hilbert Curve Region Counts (2D)")
plt.xlabel("Region X")
plt.ylabel("Region Y")
plt.show()
#"""
