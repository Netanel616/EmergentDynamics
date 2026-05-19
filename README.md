# EmergentDynamics: High-Performance Multi-Agent Flocking Engine

An interactive, high-performance 2D active matter simulation implementing the Vicsek Flocking Model. Built using Data-Oriented Design (DOD) principles and fully vectorized Linear Algebra in NumPy to achieve native execution speeds.

## Key Architectural & Physical Features

- **Data-Oriented Design (DOD):** Replaced slow Python object-oriented classes with flat, contiguous memory matrices (`np.ndarray`) to ensure maximum CPU L1/L2 cache locality and SIMD acceleration.
- **The Minimum Image Convention:** Pairwise displacements and distances are calculated across a closed toroidal space (periodic boundaries) using highly efficient matrix broadcasting: $\Delta X = X[:, \text{None}, :] - X[\text{None}, :, :]$.
- **Rotational Inertia & Smooth Turning:** Implemented turning attack-rate limits ($\omega$) to eliminate artificial high-frequency snapping/jittering, resulting in continuous fluid trajectories.
- **Continuous Angular Diffusion:** Scaled stochastic noise fluctuations dynamically by $\sqrt{dt}$ to model a mathematically consistent physical Brownian motion independent of the frame-rate.
- **Decoupled Update Strides:** Separated computationally expensive neighborhood updates ($O(N^2)$) from the continuous graphics rendering thread.

## Interactive Controls HUD

Run the live simulation and toggle states in real-time inside the Pygame display window:

- **SPACEBAR:** Toggle dynamically between Classic Vicsek Snapping and Smooth Inertial Motion.
- **C:** Toggle dynamically between alignment On/Off.

- **ARROW UP / DOWN (or + / -):** Interactively scale environmental noise amplitude ($\eta$).
- **W / S:** Tweak maximum turning rate speed limits on the fly.
- **H:** Show/Hide the semi-transparent Heads-Up Display (HUD) dashboard overlay.

## Project Structure

```text
├── src/
│   ├── engine.py       # Core physics and numerical integration loops
│   └── visualizer.py   # High-performance Pygame rendering pipeline (View)
├── tests/
│   └── test_engine.py  # Comprehensive automated math and boundary unit tests
├── run.py              # Main interactive entry point
├── requirements.txt    # Managed package dependencies
└── THEORY.md           # Rigorous mathematical and physical manual
```

## Quick Start & Execution

Ensure you have Python 3.8+ installed, then clone the repository and execute:

```bash
pip install -r requirements.txt
pytest                 # Run automated verification test suite
python run.py          # Launch interactive simulation
```
