import numpy as np
from src.engine import EmergentEngine
from src.visualizer import EmergentVisualizer


def main() -> None:
    # 1. Configuration parameters
    num_agents = 200  # Start with a clean population size
    domain_size = 100.0  # Physical size of space
    agent_speed = 1.5  # Constant travel speed
    target_fps = 60  # Smooth graphics rate
    time_delta = 0.15  # Simulation step size per frame

    # 2. Instantiate our decoupled components
    print("[Engine] Initializing active state matrices...")
    engine = EmergentEngine(
        num_agents=num_agents,
        domain_size=domain_size,
        speed=agent_speed
    )

    print("[Visualizer] Launching Pygame render display...")
    visualizer = EmergentVisualizer(
        engine=engine,
        window_size=800
    )

    # 3. Hand control off to the main loop
    print("[Simulation] Running. Press 'ESC' or close the window to exit.")
    visualizer.run(fps=target_fps, dt=time_delta)


if __name__ == "__main__":
    from src.visualizer import EmergentVisualizer  # Ensure import resolution

    main()