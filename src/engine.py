from typing import cast
import numpy as np


class EmergentEngine:
    """
    Core simulation engine for EmergentDynamics.
    Manages the vectorized state matrices of a multi-agent crowd system.
    """
    def __init__(self, num_agents: int,
                 domain_size: float,
                 speed: float,
                 alignment_radius: float = 5.0,
                 noise_amplitude: float = 1.5,
                 max_turn_rate: float = 4.0,
                 update_stride: int = 3):
        self.N: int = num_agents
        self.L: float = domain_size
        self.v0: float = speed
        self.R: float = alignment_radius
        self.eta: float = noise_amplitude # (eta) maximum angular noise perturbation
        self.max_turn_rate:float = max_turn_rate
        self.update_stride: int = update_stride
        self.step_count: int = 0
        self.target_headings: np.ndarray = np.zeros(self.N, dtype=np.float64)

        #Initialize the system state
        self.positions: np.ndarray = np.zeros((self.N, 2), dtype= np.float64)
        self.velocities: np.ndarray = np.zeros((self.N, 2), dtype= np.float64)
        self.headings: np.ndarray = np.zeros(self.N, dtype= np.float64)

        self.initialize_random_state()

        # Global Order Parameter
        self.phi: float = 1.0
        self.phi_history: list = []

        # Interaction Radii and Weights
        self.r_repulsion: float = 1.8
        self.w_repulsion: float = 0.85


    def initialize_random_state(self):
        """
        Distributes agents uniformly across the continuous 2D domain
        and assigns random initial heading directions.
        """
        self.positions = cast(np.ndarray, np.random.uniform(0, self.L, size=(self.N, 2)))
        self.headings = cast(np.ndarray, np.random.uniform(-np.pi, np.pi, size=self.N))
        self.target_headings = self.headings.copy()
        # Compute velocity vectir based on headings: V = [v0*cos(theta), v0*sin(theta)]
        self.update_velocities_from_headings()

    def update_velocities_from_headings(self):
        """
        Vectorized update maps 1D heading angles to 2D velocity vectors
        using trigonometric projections.
        """
        self.velocities[:, 0] = self.v0 * np.cos(self.headings)
        self.velocities[:, 1] = self.v0 * np.sin(self.headings)

    def step(self, dt: float):
        """
        Advances the simulation by a single time step dt using Euler Integration.
        Enforces periodic (toroidal) boundary conditions across the domain.

        :param dt: Time step delta (discrete time increment)
        """
        self.positions += self.velocities * dt

        # Periodic boundary conditions
        self.positions = self.positions % self.L
        self.calculate_global_order()

    def align_headings(self) -> None:
        """
        Calculates local neighborhood heading alignments in parallel.
        Uses broadcasting and the Minimum Image Convention to calculate
        toroidal distances.
        """

        diff = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]

        diff = diff - self.L * np.round(diff / self.L)

        dist_sq = np.sum(diff ** 2, axis=-1)

        neighbors_mask = dist_sq < (self.R ** 2)

        repulsion_mask = (dist_sq < (self.r_repulsion ** 2)) & (dist_sq > 0.0)

        sin_headings = np.sin(self.headings)
        cos_headings = np.cos(self.headings)

        sin_sum = np.sum(sin_headings[np.newaxis, :] * neighbors_mask, axis=1)
        cos_sum = np.sum(cos_headings[np.newaxis, :] * neighbors_mask, axis=1)
        # Normalize the accumulated alignment vectors
        align_magnitude = np.sqrt(sin_sum ** 2 + cos_sum ** 2)
        align_vectors = np.zeros((self.N, 2))
        valid_align = align_magnitude > 0
        align_vectors[valid_align, 0] = cos_sum[valid_align] / align_magnitude[valid_align]
        align_vectors[valid_align, 1] = sin_sum[valid_align] / align_magnitude[valid_align]

        # --- PART B: Short-range collision avoidance ---
        dist = np.sqrt(dist_sq)

        # diff is (pos_j - pos_i). To steer away from j, we must move along -diff
        direction_vectors = diff / (dist[:, :, np.newaxis] + 1e-9)

        # Accumulate and normalize repulsion forces
        total_repulsion = np.sum(direction_vectors * repulsion_mask[:, :, np.newaxis], axis=1)
        repulsion_magnitude = np.linalg.norm(total_repulsion, axis=1)

        repulse_vectors = np.zeros((self.N, 2))
        valid_repulse = repulsion_magnitude > 0
        repulse_vectors[valid_repulse] = total_repulsion[valid_repulse] / repulsion_magnitude[valid_repulse][
            :, np.newaxis]

        # --- PART C: Vector blending and target assignment ---
        has_repulsion = repulsion_magnitude > 0
        combined_vectors = np.where(
            has_repulsion[:, np.newaxis],
            (1.0 - self.w_repulsion) * align_vectors + self.w_repulsion * repulse_vectors,
            align_vectors
        )

        # Update target headings if any neighbors are detected in either zone
        has_any_neighbors = (np.sum(neighbors_mask, axis=1) > 0) | has_repulsion
        self.target_headings = np.where(
            has_any_neighbors,
            np.arctan2(combined_vectors[:, 1], combined_vectors[:, 0]),
            self.target_headings
        )


    def step_with_alignment(self, dt: float) -> None:
        """
        Advances the simulation by a single time step dt using Euler Integration.
        Calculates local alignments and enforces periodic boundary conditions.

        :param dt: Time step delta (discrete time increment)
        """
        # Align headings of neighboring agents
        self.align_headings()
        self.headings = self.target_headings.copy()
        # noise
        self.apply_absolute_noise()
        # Keep velocities in sync with newly aligned headings
        self.update_velocities_from_headings()
        # Update positions using Euler Integration
        self.positions += self.velocities * dt
        # Enforce periodic boundary wrapping
        self.positions = self.positions % self.L
        self.calculate_global_order()

    def apply_absolute_noise(self) -> None:
        """
        Adds a random, uniform angular noise perturbation within the range
        [-eta/2, eta/2] to all heading angles. Wraps angles back to [-pi, pi].
        """
        if self.eta == 0.0:
            return

        # Generate uniform random noise for all N agents simultaneously
        noise = np.random.uniform(-self.eta / 2.0, self.eta / 2.0, size=self.N)
        self.headings += noise

        # Keep angles normalized within the physical range [-pi, pi]
        # Using a continuous symmetric wrapping formula: (theta + pi) % (2*pi) - pi
        self.headings = (self.headings + np.pi) % (2.0 * np.pi) - np.pi

    def set_noise_amplitude(self, new_eta: float) -> None:
        """
        Dynamically adjusts the noise amplitude (eta) during runtime,
        clamping the value to a minimum of 0.0 (no noise).
        """
        self.eta = max(0.0, new_eta)

    def step_with_smooth_alignment(self, dt: float) -> None:
        """
        The continuous smooth physics loop: Interpolates headings towards targets
        clamped by self.max_turn_rate, decouples neighborhood planning via update strides,
        and applies continuous stochastic diffusion noise.
        """
        # Decouple spatial neighborhood planning
        if self.step_count % self.update_stride == 0:
            self.align_headings()
        self.step_count += 1

        # Turning attack rate/rotational inertia
        # Calculate the shortest angular difference to target headings
        angular_diff = (self.target_headings - self.headings + np.pi) % (2.0 * np.pi) - np.pi

        # Clamp turning step to maximum rotational speed (radians/second)
        max_step = self.max_turn_rate * dt
        turn_step = np.clip(angular_diff, -max_step, max_step)
        self.headings += turn_step

        # Apply stochastic continuous angular diffusion
        self.apply_smooth_noise(dt)

        # Synchronize velocity coordinates and integrate position state
        self.update_velocities_from_headings()
        self.positions += self.velocities * dt
        self.positions = self.positions % self.L
        self.calculate_global_order()


    def apply_smooth_noise(self, dt: float) -> None:
        """
        Stochastic physics noise: Scales random noise by sqrt(dt) to guarantee
        that the perturbation models a mathematically consistent continuous
        diffusion process (Wiener process) independent of frame-rate.
        """
        if self.eta == 0.0:
            return
        # Scaling by sqrt(dt) keeps angular diffusion rates stable at variable FPS
        noise = np.random.uniform(-self.eta / 2.0, self.eta / 2.0, size=self.N) * np.sqrt(dt)
        self.headings += noise
        self.headings = (self.headings + np.pi) % (2.0 * np.pi) - np.pi

    def calculate_global_order(self) -> None:
        """
        Calculates the global order parameter (phi), which measures the system's
        total polarization. Returns a value between 0.0 (complete chaos/isotropic)
        and 1.0 (perfect global alignment).
        """
        total_velocity_vector = np.sum(self.velocities, axis=0)

        # Calculate the Euclidean magnitude (norm) of the accumulated vector
        net_magnitude = np.linalg.norm(total_velocity_vector)

        # Normalize by the theoretical maximum possible magnitude (N * v0)
        phi = net_magnitude / (self.N * self.v0)

        self.phi = float(phi)
