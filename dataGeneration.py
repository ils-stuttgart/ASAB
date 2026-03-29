import numpy as np


def generate_realistic_cluster(center, n_points=50, spread=0.05, random_state=42):
    rng = np.random.default_rng(random_state)
    vectors = rng.normal(loc=center, scale=spread, size=(n_points, len(center)))
    centroid = np.mean(vectors, axis=0)
    return vectors, centroid


def generate_multiple_clusters():
    clusters = []

    centers = [
        [0.2, 0.2],
        [0.7, 0.3],
        [0.4, 0.8],
    ]

    spreads = [0.03, 0.05, 0.04]

    for i, (center, spread) in enumerate(zip(centers, spreads)):
        vectors, centroid = generate_realistic_cluster(
            center=center,
            n_points=60,
            spread=spread,
            random_state=42 + i,
        )
        clusters.append((vectors, centroid))

    return clusters