import numpy as np


class EmergentEngine:
    """
    Core simulation engine for EmergentDynamics.
    Manages the vectorized state matrices of a multi-agent crowd system.
    """

    def __init__(self, num_agents: int, domain_size: float, speed: float):
        self.N: int = num_agents
        self.L: float = domain_size
        self.v0: float = speed

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

