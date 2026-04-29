import numpy as np
import pytest

from ASAB.MonteCarlo.MonteCarlo import (
    Cluster,
    monte_carlo_sample_cluster,
    monte_carlo_sample_multiple_clusters,
    sample_from_epsilon_ball,
)


def make_simple_cluster():
    vectors = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
    ])
    centroid = np.array([0.0, 0.0])
    return Cluster(vectors=vectors, centroid=centroid)


def test_cluster_init_valid():
    cluster = make_simple_cluster()
    assert cluster.vectors.shape == (4, 2)
    assert cluster.centroid.shape == (2,)


def test_cluster_dimension_mismatch():
    vectors = np.array([[1.0, 2.0], [3.0, 4.0]])
    centroid = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        Cluster(vectors=vectors, centroid=centroid)


def test_sample_from_epsilon_ball_shape():
    centroid = np.array([0.0, 0.0, 0.0])
    samples = sample_from_epsilon_ball(centroid, radius=1.0, n_samples=50, random_state=42)
    assert samples.shape == (50, 3)


def test_sample_from_epsilon_ball_zero_radius():
    centroid = np.array([1.0, 2.0])
    samples = sample_from_epsilon_ball(centroid, radius=0.0, n_samples=10, random_state=42)
    expected = np.tile(centroid, (10, 1))
    assert np.allclose(samples, expected)


def test_monte_carlo_sample_cluster_shape():
    cluster = make_simple_cluster()
    samples = monte_carlo_sample_cluster(cluster, epsilon=0.5, n_samples=100, random_state=42)
    assert samples.shape == (100, 2)


def test_monte_carlo_sample_cluster_epsilon_zero():
    cluster = make_simple_cluster()
    samples = monte_carlo_sample_cluster(cluster, epsilon=0.0, n_samples=20, random_state=42)
    expected = np.tile(cluster.centroid, (20, 1))
    assert np.allclose(samples, expected)


def test_monte_carlo_sample_cluster_within_radius():
    cluster = make_simple_cluster()
    epsilon = 0.5
    samples = monte_carlo_sample_cluster(cluster, epsilon=epsilon, n_samples=500, random_state=42)

    distances = np.linalg.norm(samples - cluster.centroid, axis=1)

    # In this cluster, max distance from centroid is 1.0
    assert np.all(distances <= epsilon * 1.0 + 1e-10)


def test_invalid_epsilon_low():
    cluster = make_simple_cluster()
    with pytest.raises(ValueError):
        monte_carlo_sample_cluster(cluster, epsilon=-0.1, n_samples=10)


def test_invalid_epsilon_high():
    cluster = make_simple_cluster()
    with pytest.raises(ValueError):
        monte_carlo_sample_cluster(cluster, epsilon=1.5, n_samples=10)


def test_invalid_n_samples():
    cluster = make_simple_cluster()
    with pytest.raises(ValueError):
        monte_carlo_sample_cluster(cluster, epsilon=0.5, n_samples=0)


def test_reproducibility():
    cluster = make_simple_cluster()

    s1 = monte_carlo_sample_cluster(cluster, epsilon=0.8, n_samples=100, random_state=123)
    s2 = monte_carlo_sample_cluster(cluster, epsilon=0.8, n_samples=100, random_state=123)

    assert np.allclose(s1, s2)


def test_multiple_clusters():
    cluster1 = make_simple_cluster()

    vectors2 = np.array([
        [2.0, 2.0],
        [3.0, 2.0],
        [2.0, 3.0],
    ])
    centroid2 = np.array([2.33333333, 2.33333333])
    cluster2 = Cluster(vectors=vectors2, centroid=centroid2)

    results = monte_carlo_sample_multiple_clusters(
        [cluster1, cluster2],
        epsilon=0.5,
        n_samples_per_cluster=25,
        random_state=42,
    )

    assert len(results) == 2
    assert results[0].shape == (25, 2)
    assert results[1].shape == (25, 2)