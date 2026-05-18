import numpy as np
from src.engine import EmergentEngine


def test_engine_initialization_dimensions() -> None:
    """
    Verify that the state matrices are allocated with the correct structural shapes.
    """
    num_agents = 100
    domain_size = 50.0
    speed = 2.0

    engine = EmergentEngine(num_agents=num_agents, domain_size=domain_size, speed=speed)

    # Assert matrix dimensions
    assert engine.positions.shape == (num_agents, 2)
    assert engine.velocities.shape == (num_agents, 2)
    assert engine.headings.shape == (num_agents,)


def test_engine_initialization_bounds() -> None:
    """
    Verify that agents are distributed strictly within the boundaries of the domain
    and angles are within physical bounds [-pi, pi].
    """
    num_agents = 500
    domain_size = 10.0
    speed = 1.0

    engine = EmergentEngine(num_agents=num_agents, domain_size=domain_size, speed=speed)

    # Check spatial boundaries
    assert np.all(engine.positions >= 0.0)
    assert np.all(engine.positions <= domain_size)

    # Check angular boundaries
    assert np.all(engine.headings >= -np.pi)
    assert np.all(engine.headings <= np.pi)


def test_vector_magnitude_matches_speed() -> None:
    """
    Verify that the calculated velocity vectors have a Euclidean magnitude
    exactly equal to the constant speed scalar (v0).
    """
    num_agents = 50
    domain_size = 100.0
    speed = 3.5

    engine = EmergentEngine(num_agents=num_agents, domain_size=domain_size, speed=speed)

    # Calculate Euclidean norm (magnitude) of velocity vectors along columns (axis=1)
    # Magnitude = sqrt(vx^2 + vy^2)
    magnitudes = np.linalg.norm(engine.velocities, axis=1)

    # Verify magnitudes equal speed (allowing for minor floating-point tolerances via np.allclose)
    assert np.allclose(magnitudes, speed)