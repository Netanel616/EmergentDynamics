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
                 update_stride: int = 3,
                 fast_on: bool = False):
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

        # Spatial Hashing Grid Settings
        self.grid_cell_size: float = self.R
        self.grid_num_cells_per_axis: int = int(np.floor(self.L / self.grid_cell_size))

        # Algorithmic Mode Flag (True = O(N) Spatial Hashing, False = O(N^2) Bruteforce)
        self.use_fast_compute: bool = fast_on


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
        if self.use_fast_compute:
            self.fast_align_headings()
        else:
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
            if self.use_fast_compute:
                self.fast_align_headings()
            else:
                self.align_headings()
        self.step_count+=1

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

    def fast_align_headings(self) -> None:
        """
        Calculates heading alignments by evaluating local blocks cell-by-cell.
        Reduces complexity by breaking down grid building and block processing
        into distinct, readable sub-functions.
        """
        if self.N <= 1:
            return

        # 1. Build the grid partition structure
        grid_buckets, _ = self._build_spatial_grid()
        new_targets = self.target_headings.copy()

        # 2. Iterate through each cell block and compute localized interactions
        for cell_idx, agents_in_cell in grid_buckets.items():
            if not agents_in_cell:
                continue

            self._process_cell_block(cell_idx, agents_in_cell, grid_buckets, new_targets)

        self.target_headings = new_targets

    def _build_spatial_grid(self) -> tuple[dict, np.ndarray]:
        """
        Partitions the domain space into discrete coordinate buckets.
        Returns a dictionary mapping cell indices to agent lists, and the coordinate matrix.
        """
        K = self.grid_num_cells_per_axis
        cell_coords = np.floor(self.positions / self.grid_cell_size).astype(np.int32)
        cell_coords = np.clip(cell_coords, 0, K - 1)

        grid_buckets = {c: [] for c in range(K * K)}
        for agent_idx in range(self.N):
            col, row = cell_coords[agent_idx]
            cell_1d_idx = col + row * K
            grid_buckets[cell_1d_idx].append(agent_idx)

        return grid_buckets, cell_coords

    def _get_cell_neighborhood_candidates(self, cell_idx: int, grid_buckets: dict) -> list[int]:
        """
        Gathers all agent indices residing within the 3x3 toroidal neighborhood of a cell.
        """
        K = self.grid_num_cells_per_axis
        col = cell_idx % K
        row = cell_idx // K

        neighborhood_candidates = []
        for d_col in [-1, 0, 1]:
            for d_row in [-1, 0, 1]:
                n_col = (col + d_col) % K
                n_row = (row + d_row) % K
                n_cell_1d = n_col + n_row * K
                neighborhood_candidates.extend(grid_buckets[n_cell_1d])

        return neighborhood_candidates

    def _compute_block_geometry(self, cell_agents: list[int], candidates: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """
        Executes broadcasting matrix operations to find relative displacements
        and squared distances under the Minimum Image Convention.
        """
        cell_agents_arr = np.array(cell_agents)
        candidates_arr = np.array(candidates)

        # Broadcasting shapes: (Num Agents in Cell, Num Candidates in Neighborhood, 2)
        diff = self.positions[cell_agents_arr, np.newaxis, :] - self.positions[np.newaxis, candidates_arr, :]
        diff = diff - self.L * np.round(diff / self.L)
        dist_sq = np.sum(diff ** 2, axis=-1)

        return diff, dist_sq

    def _calculate_agent_forces(self, idx: int, agent_align_mask: np.ndarray, agent_repulse_mask: np.ndarray,
                                diff_slice: np.ndarray, dist_sq_slice: np.ndarray,
                                sin_headings: np.ndarray, cos_headings: np.ndarray) -> tuple[
        np.ndarray, np.ndarray, bool]:
        """
        Calculates the distinct local alignment and short-range repulsion unit vectors
        for a single agent within the block.
        """
        # --- PART A: Alignment Vector ---
        if np.any(agent_align_mask):
            sin_sum = np.sum(sin_headings * agent_align_mask)
            cos_sum = np.sum(cos_headings * agent_align_mask)
            align_mag = np.sqrt(sin_sum ** 2 + cos_sum ** 2)
            align_vec = np.array([cos_sum, sin_sum]) / (align_mag + 1e-9) if align_mag > 0 else np.zeros(2)
        else:
            align_vec = np.zeros(2)

        # --- PART B: Repulsion Vector ---
        if np.any(agent_repulse_mask):
            dist = np.sqrt(dist_sq_slice[agent_repulse_mask])
            dir_vectors = diff_slice[agent_repulse_mask] / (dist[:, np.newaxis] + 1e-9)
            total_repulsion = np.sum(dir_vectors, axis=0)
            repulsion_mag = np.linalg.norm(total_repulsion)
            repulse_vec = total_repulsion / repulsion_mag if repulsion_mag > 0 else np.zeros(2)
            has_repulsion = True
        else:
            repulse_vec = np.zeros(2)
            has_repulsion = False

        return align_vec, repulse_vec, has_repulsion

    def _blend_and_update_heading(self, agent_idx: int, has_repulsion: bool, any_alignment: bool,
                                  align_vec: np.ndarray, repulse_vec: np.ndarray, new_targets: np.ndarray) -> None:
        """
        Blends the calculated physical forces using weight priorities and commits
        the final resolved angle to the target array.
        """
        if has_repulsion:
            combined_vec = (1.0 - self.w_repulsion) * align_vec + self.w_repulsion * repulse_vec
        elif any_alignment:
            combined_vec = align_vec
        else:
            return

        if np.linalg.norm(combined_vec) > 0:
            new_targets[agent_idx] = np.arctan2(combined_vec[1], combined_vec[0])

    def _process_cell_block(self, cell_idx: int, agents_in_cell: list[int],
                            grid_buckets: dict, new_targets: np.ndarray) -> None:
        """
        Orchestrates localized matrix math for a single cell by coordinating
        geometry computation, force calculation, and heading updates.
        """
        candidates = self._get_cell_neighborhood_candidates(cell_idx, grid_buckets)
        if not candidates:
            return

        # 1. Compute vectorized block geometry
        diff, dist_sq = self._compute_block_geometry(agents_in_cell, candidates)

        # 2. Extract logical interaction masks (excluding self-interaction)
        align_mask = (dist_sq < (self.R ** 2)) & (dist_sq > 0.0)
        repulsion_mask = (dist_sq < (self.r_repulsion ** 2)) & (dist_sq > 0.0)

        candidates_arr = np.array(candidates)
        sin_headings = np.sin(self.headings[candidates_arr])
        cos_headings = np.cos(self.headings[candidates_arr])

        # 3. Process each individual agent within this spatial block
        for idx, agent_i in enumerate(agents_in_cell):
            align_vec, repulse_vec, has_repulsion = self._calculate_agent_forces(
                idx, align_mask[idx], repulsion_mask[idx], diff[idx], dist_sq[idx], sin_headings, cos_headings
            )

            self._blend_and_update_heading(
                agent_i, has_repulsion, np.any(align_mask[idx]), align_vec, repulse_vec, new_targets
            )
