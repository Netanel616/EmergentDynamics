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


def test_toroidal_boundary_wrapping() -> None:
    """
    Verify that agents wrapping past the domain boundaries correctly
    teleport to the opposite side of the toroidal space.
    """
    num_agents = 2
    domain_size = 10.0
    speed = 1.0

    engine = EmergentEngine(num_agents=num_agents, domain_size=domain_size, speed=speed)

    # Force specific initial positions near the boundaries
    # Agent 0: Near the right boundary (x = 9.5, y = 5.0)
    # Agent 1: Near the left boundary (x = 0.5, y = 5.0)
    engine.positions = np.array([
        [9.5, 5.0],
        [0.5, 5.0]
    ], dtype=np.float64)

    # Force explicit velocities
    # Agent 0: Moving right (+x) -> will cross the right boundary
    # Agent 1: Moving left (-x) -> will cross the left boundary
    engine.velocities = np.array([
        [1.0, 0.0],
        [-1.0, 0.0]
    ], dtype=np.float64)

    # Advance the simulation by dt = 1.0 time units
    # Expected positions before wrapping:
    # Agent 0: 9.5 + 1.0 = 10.5  --> Should wrap to 10.5 % 10.0 = 0.5
    # Agent 1: 0.5 - 1.0 = -0.5  --> Should wrap to -0.5 % 10.0 = 9.5
    dt = 1.0
    engine.step(dt)

    expected_positions = np.array([
        [0.5, 5.0],
        [9.5, 5.0]
    ], dtype=np.float64)

    # Assert that actual wrapped positions match expected positions within tolerance
    assert np.allclose(engine.positions, expected_positions)