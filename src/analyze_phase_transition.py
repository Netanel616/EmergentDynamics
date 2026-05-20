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


if __name__ == "__main__":
    run_phase_transition_analysis()