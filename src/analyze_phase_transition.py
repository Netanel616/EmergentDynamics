import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.engine import EmergentEngine


def run_phase_transition_analysis():
    """
    Runs the flocking engine across a spectrum of noise values (eta) without
    GUI rendering to systematically chart the thermodynamic phase transition.
    """
    print("[Analysis] Starting phase transition simulation sweep...")

    # Simulation configuration
    num_agents = 300
    domain_size = 50.0
    speed = 1.0
    alignment_radius = 5.0

    # Sweep setup: 15 linear steps of noise from 0.0 up to 5.0 rad
    noise_steps = np.linspace(0.0, 7.0, num=150)
    total_steps = 400
    burn_in_steps = 250  # Allow the system to stabilize before recording data

    results = []

    # Iterate through each environmental noise amplitude step
    for eta in noise_steps:
        print(f" --> Evaluating Noise Level (eta) = {eta:.2f} rad")

        # Instantiate a clean engine for this specific temperature state
        engine = EmergentEngine(
            num_agents=num_agents,
            domain_size=domain_size,
            speed=speed,
            alignment_radius=alignment_radius,
            noise_amplitude=eta
        )

        steady_state_orders = []

        # Fast non-graphical simulation loop
        for step in range(total_steps):
            engine.step_with_alignment(dt=0.15)

            # Record metrics only AFTER the burn-in window to eliminate transient states
            if step >= burn_in_steps:
                steady_state_orders.append(engine.phi)

        # Calculate the mean global order for this noise profile
        mean_order = np.mean(steady_state_orders)
        results.append({"noise": eta, "order_parameter": mean_order})

    # Save metrics to a CSV file for future logging or verification
    df = pd.DataFrame(results)
    os.makedirs("data", exist_ok=True)
    csv_path = "data/phase_transition_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Analysis] Core metrics successfully stored in: {csv_path}")

    # Generate the professional scientific plot
    plt.figure(figsize=(8, 5.5))

    # Plot data points and continuous curve
    plt.plot(df['noise'], df['order_parameter'], marker='o', linestyle='-', color='#10b981', linewidth=2,
             label=r'Simulation Data ($\phi$)')

    # Adjust layout and scientific labels
    plt.title("Vicsek Flocking Model - Phase Transition Analysis", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(r"Environmental Noise Amplitude ($\eta$) [radians]", fontsize=12)
    plt.ylabel(r"Global Order Parameter ($\phi$)", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.xlim(-0.1, 5.2)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    # Export visualization
    plot_path = "data/phase_transition_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"[Analysis] Scientific plot generated and saved to: {plot_path}")
    plt.show()


def run_hysteresis_analysis():
    """
    Executes a continuous thermodynamic forward (heating) and backward (cooling)
    sweep on a single engine instance to empirically chart the system's hysteresis loop.
    """
    print("[Hysteresis Analysis] Initializing thermodynamic continuous sweep...")

    # Core configuration
    num_agents = 500
    domain_size = 200.0
    speed = 1.2
    alignment_radius = 5.0

    # Define continuous steps for noise amplitude (0.0 to 4.5 rad)
    steps = 45
    noise_profiles = np.linspace(0.0, 4.5, num=steps)

    steps_per_state = 300
    burn_in_steps = 250  # Let the system adapt to the new noise level before sampling

    # Instantiate ONE single engine to maintain history/spatial memory across states
    engine = EmergentEngine(
        num_agents=num_agents,
        domain_size=domain_size,
        speed=speed,
        alignment_radius=alignment_radius,
        noise_amplitude=0.0,
        fast_on = True
    )

    forward_results = []

    # --- PHASE 1: Forward Sweep (Heating up the system) ---
    print(" --> Starting Forward Sweep (Increasing Noise)...")
    for eta in noise_profiles:
        engine.eta = eta  # Mutate environmental noise inline without resetting positions
        state_orders = []

        for step in range(steps_per_state):
            engine.step_with_alignment(dt=0.15)
            if step >= burn_in_steps:
                state_orders.append(engine.phi)

        forward_results.append({"noise": eta, "order_parameter": np.mean(state_orders)})

    backward_results = []

    # --- PHASE 2: Backward Sweep (Cooling down the system) ---
    print(" --> Starting Backward Sweep (Decreasing Noise)...")
    for eta in reversed(noise_profiles):
        engine.eta = eta  # Mutate environmental noise down inline
        state_orders = []

        for step in range(steps_per_state):
            engine.step_with_alignment(dt=0.15)
            if step >= burn_in_steps:
                state_orders.append(engine.phi)

        backward_results.append({"noise": eta, "order_parameter": np.mean(state_orders)})

    # Organize data into dataframes for unified logging
    df_forward = pd.DataFrame(forward_results)
    df_backward = pd.DataFrame(backward_results)

    # Save datasets
    os.makedirs("data", exist_ok=True)
    df_forward.to_csv("data/hysteresis_forward.csv", index=False)
    df_backward.to_csv("data/hysteresis_backward.csv", index=False)

    # --- PLOTTING ---
    plt.figure(figsize=(8.5, 6))

    # Plot Forward Sweep (Heating)
    plt.plot(df_forward['noise'], df_forward['order_parameter'],
             marker='o', linestyle='-', color='#ef4444', linewidth=2, label='Forward Sweep (Heating)')

    # Plot Backward Sweep (Cooling)
    plt.plot(df_backward['noise'], df_backward['order_parameter'],
             marker='s', linestyle='--', color='#3b82f6', linewidth=2, label='Backward Sweep (Cooling)')

    plt.title("Vicsek Model - Hysteresis Loop Analysis", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(r"Environmental Noise Amplitude ($\eta$) [radians]", fontsize=12)
    plt.ylabel(r"Global Order Parameter ($\phi$)", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.xlim(-0.1, 4.7)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    plot_path = "data/hysteresis_loop_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"[Analysis] Hysteresis plot saved successfully to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    #run_phase_transition_analysis()
    run_hysteresis_analysis()