#from ASAB.MonteCarlo.MonteCarlo import Cluster, monte_carlo_sample_cluster
from dataGeneration import generate_realistic_cluster
#from ASAB.MonteCarlo.visualization import visualize_cluster_and_samples
from MonteCarlo.visualization import visualize_cluster_and_samples
from config import load_config

from MonteCarlo.MonteCarlo import Cluster, monte_carlo_sample_cluster

config = load_config()

epsilon = config["monte_carlo"]["epsilon"]
n_samples = config["monte_carlo"]["n_samples"]

# Generate cluster
vectors, centroid = generate_realistic_cluster(
    center=[0.5, 0.5],
    n_points=100,
    spread=0.05,
)

cluster = Cluster(vectors=vectors, centroid=centroid)

# Sample
samples = monte_carlo_sample_cluster(
    cluster=cluster,
    epsilon=epsilon,
    n_samples=n_samples,
    random_state=42,
)

# Visualize
visualize_cluster_and_samples(cluster, samples, epsilon)