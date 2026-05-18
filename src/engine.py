import numpy as np


class EmergantEngine:
    """
    Core simulation engine for EmergentDynamics.
    Manages the vectorized state matrices of a multi-agent crowd system.
    """

    def __init__(self, num_agent: int, domain_size: float, speed: float):
        self.N: int = num_agent
        self.L: float = domain_size
        self.v0: float = speed

        #Initialize the system state
        self.positions: np.ndarray = np.zero((self.N, 2), dtype= np.float64)
        self.velocities: np.ndarray = np.zero((self.N, 2), dtype= np.float64)
        self.heading: np.ndarray = np.zero(self.N, dtype= np.float64)

        self.initialize_random_state()

    def initialize_random_state(self):
        """
        Distributes agents uniformly across the continuous 2D domain
        and assigns random initial heading directions.
        """
        # Uniform spatial distribution: position between 0 and L
        self.positions = np.random.uniform(0, self.L, size=(self.N, 2))

        # Uniform direction distribution: angels between -pi and pi
        self.heading = np.random.uniform(-np.pi, np.pi, size=self.N)

        # Compute velocity vectir based on headings: V = [v0*cos(theta), v0*sin(theta)]
        self.update_velocities_from_headings()

    def update_velocities_from_headings(self):
        """
        Vectorized update maps 1D heading angles to 2D velocity vectors
        using trigonometric projections.
        """
        self.velocities[:, 0] = self.v0 * np.cos(self.heading)
        self.velocities[:, 1] = self.v0 * np.sin(self.heading)
