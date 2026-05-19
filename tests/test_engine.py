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


def test_step_with_alignment_basic() -> None:
    """
    Verify that step_with_alignment correctly aligns neighboring agents
    within the interaction radius while advancing their positions.
    """
    num_agents = 3
    domain_size = 20.0
    speed = 1.0
    alignment_radius = 3.0

    engine = EmergentEngine(
        num_agents=num_agents,
        domain_size=domain_size,
        speed=speed,
        alignment_radius=alignment_radius
    )

    # Place Agent 0 and Agent 1 close together (distance = 1.0 < radius = 3.0)
    # Place Agent 2 far away (distance > radius = 3.0)
    engine.positions = np.array([
        [10.0, 10.0],
        [11.0, 10.0],
        [18.0, 18.0]
    ], dtype=np.float64)

    # Headings: Agent 0 (pi/4), Agent 1 (-pi/4), Agent 2 (pi/2)
    engine.headings = np.array([np.pi / 4, -np.pi / 4, np.pi / 2], dtype=np.float64)
    engine.update_velocities_from_headings()

    # Expected alignment calculations:
    # Agent 0 & 1 are neighbors. Average components:
    # cos_avg = (cos(pi/4) + cos(-pi/4))/2 = (0.707 + 0.707)/2 = 0.707
    # sin_avg = (sin(pi/4) + sin(-pi/4))/2 = (0.707 - 0.707)/2 = 0.0
    # Aligned heading = atan2(0.0, 0.707) = 0.0 rad
    #
    # Agent 2 is isolated, so its heading remains pi/2
    expected_headings_after_alignment = np.array([0.0, 0.0, np.pi / 2], dtype=np.float64)

    dt = 1.0
    # Run the new step method with integrated alignment
    engine.step_with_alignment(dt)

    # 1. Assert headings aligned correctly
    assert np.allclose(engine.headings, expected_headings_after_alignment)

    # 2. Assert velocities updated to match new aligned headings
    expected_velocities = np.array([
        [1.0, 0.0],  # cos(0), sin(0)
        [1.0, 0.0],  # cos(0), sin(0)
        [0.0, 1.0]  # cos(pi/2), sin(pi/2) -> rounding handles floating point
    ], dtype=np.float64)
    assert np.allclose(engine.velocities, expected_velocities, atol=1e-7)

    # 3. Assert positions updated based on aligned velocities
    # Note: Physics update uses the newly computed velocities during the step
    expected_positions = np.array([
        [11.0, 10.0],  # 10.0 + 1.0 * dt
        [12.0, 10.0],  # 11.0 + 1.0 * dt
        [18.0, 19.0]  # 18.0 + 1.0 * dt
    ], dtype=np.float64)
    assert np.allclose(engine.positions, expected_positions, atol=1e-7)


def test_step_with_alignment_toroidal() -> None:
    """
    Verify that agents near opposite boundaries wrap correctly and align their
    headings using the Minimum Image Convention within step_with_alignment.
    """
    num_agents = 2
    domain_size = 10.0
    speed = 1.0
    alignment_radius = 2.0

    engine = EmergentEngine(
        num_agents=num_agents,
        domain_size=domain_size,
        speed=speed,
        alignment_radius=alignment_radius
    )

    # Place Agent 0 at left edge (0.5) and Agent 1 at right edge (9.5)
    # Wrapped toroidal distance = 1.0 (within alignment radius of 2.0)
    engine.positions = np.array([
        [0.5, 5.0],
        [9.5, 5.0]
    ], dtype=np.float64)

    # Orthogonal headings: 0.0 and pi/2. Average heading is pi/4.
    engine.headings = np.array([0.0, np.pi / 2], dtype=np.float64)
    engine.update_velocities_from_headings()

    dt = 1.0
    engine.step_with_alignment(dt)

    # Verify headings successfully aligned over wrapped space
    expected_headings = np.array([np.pi / 4, np.pi / 4], dtype=np.float64)
    assert np.allclose(engine.headings, expected_headings)

    # Verify positions wrapped correctly with new aligned velocities
    # vx = cos(pi/4) ≈ 0.7071, vy = sin(pi/4) ≈ 0.7071
    # Agent 0: [0.5 + 0.7071, 5.0 + 0.7071] = [1.2071, 5.7071]
    # Agent 1: [9.5 + 0.7071, 5.0 + 0.7071] = [10.2071, 5.7071] -> wraps to [0.2071, 5.7071]
    expected_positions = np.array([
        [1.20710678, 5.70710678],
        [0.20710678, 5.70710678]
    ], dtype=np.float64)

    assert np.allclose(engine.positions, expected_positions)