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
                 noise_amplitude: float = 0.0):
        self.N: int = num_agents
        self.L: float = domain_size
        self.v0: float = speed
        self.R = alignment_radius
        self.eta = noise_amplitude # (eta) maximum angular noise perturbation

        #Initialize the system state
        self.positions: np.ndarray = np.zeros((self.N, 2), dtype= np.float64)
        self.velocities: np.ndarray = np.zeros((self.N, 2), dtype= np.float64)
        self.headings: np.ndarray = np.zeros(self.N, dtype= np.float64)

        self.initialize_random_state()

    def initialize_random_state(self):
        """
        Distributes agents uniformly across the continuous 2D domain
        and assigns random initial heading directions.
        """
        # Uniform spatial distribution: position between 0 and L
        self.positions = np.random.uniform(0, self.L, size=(self.N, 2))

        # Uniform direction distribution: angels between -pi and pi
        self.headings = np.random.uniform(-np.pi, np.pi, size=self.N)

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
        # matrix operation: (N,2) = (N,2) + (N,2) * dt
        self.positions += self.velocities * dt

        # Periodic boundary conditions
        self.positions = self.positions % self.L

    def align_headings(self) -> None:
        """
        Calculates local neighborhood heading alignments in parallel.
        Uses broadcasting and the Minimum Image Convention to calculate
        toroidal distances.
        """
        # 1. Compute pairwise difference vectors using broadcasting: shape (N, N, 2)
        diff = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]

        # 2. Apply Minimum Image Convention to account for toroidal wrapping
        diff = diff - self.L * np.round(diff / self.L)

        # 3. Calculate squared Euclidean distances: shape (N, N)
        dist_sq = np.sum(diff ** 2, axis=-1)

        # 4. Create boolean neighborhood adjacency matrix: shape (N, N)
        neighbors_mask = dist_sq < (self.R ** 2)

        # 5. Extract heading vector components for all agents
        sin_headings = np.sin(self.headings)
        cos_headings = np.cos(self.headings)

        # 6. Sum neighbor heading vectors using mask multiplication: shape (N,)
        # Multiplying 1D sin_headings[np.newaxis, :] (1, N) with mask (N, N)
        # averages components across neighbors (axis=1)
        sin_sum = np.sum(sin_headings[np.newaxis, :] * neighbors_mask, axis=1)
        cos_sum = np.sum(cos_headings[np.newaxis, :] * neighbors_mask, axis=1)

        # 7. Update headings using trigonometric arc tangent
        self.headings = np.arctan2(sin_sum, cos_sum)

    def step_with_alignment(self, dt: float) -> None:
        """
        Advances the simulation by a single time step dt using Euler Integration.
        Calculates local alignments and enforces periodic boundary conditions.

        :param dt: Time step delta (discrete time increment)
        """
        # noise
        self.apply_noise()

        # Align headings of neighboring agents
        self.align_headings()



        # Keep velocities in sync with newly aligned headings
        self.update_velocities_from_headings()

        # Update positions using Euler Integration
        self.positions += self.velocities * dt

        # Enforce periodic boundary wrapping
        self.positions = self.positions % self.L

    def apply_noise(self) -> None:
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

