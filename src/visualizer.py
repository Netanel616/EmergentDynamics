import sys
import pygame
from src.engine import EmergentEngine

BALL_SIZE = 4
ARROW_SIZE = 6.0

class EmergentVisualizer:
    """
    Handles the high-performance 2D rendering of the EmergentEngine state
    using the Pygame graphic library.
    """

    def __init__(self, engine: EmergentEngine, window_size: int = 800):
        self.engine: EmergentEngine = engine
        self.window_size: int = window_size

        # Calculate coordinate scaling factors to map simulation space [0, L]
        # to pixel screen coordinates [0, window_size]
        self.scale: float = self.window_size / self.engine.L

        # Initialize core Pygame systems
        pygame.init()
        pygame.display.set_caption("EmergentDynamics - Multi-Agent Simulation")

        # Allocate screen surface
        self.screen: pygame.Surface = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock: pygame.time.Clock = pygame.time.Clock()

        # Visual color palette (Teal and Amber theme)
        self.COLOR_BG = (28, 33, 41)  # Dark slate background
        self.COLOR_AGENT = (16, 185, 129)  # Vibrant Emerald green for agents
        self.COLOR_HEADING = (245, 158, 11)  # Amber vectors to show directions

    def handle_events(self) -> bool:
        """
        Processes OS windows events.
        Returns False if the window is closed, indicating the simulation must stop.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            # Allow exiting via the Escape key
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return False
        return True

    def render(self) -> None:
        """
        Renders a single frame of the simulation.
        Translates raw engine matrices into scaled pixel components.
        """
        # Clear the previous frame with background color
        self.screen.fill(self.COLOR_BG)

        # Extract direct references to positions and velocities matrices
        pos_matrix = self.engine.positions
        vel_matrix = self.engine.velocities

        # Loop over each agent to draw them
        # Note: We will keep this basic loop for Step 3, but in optimized steps
        # we can draw batch arrays directly or keep N below a few thousand for Pygame.
        for i in range(self.engine.N):
            # Scale mathematical float coordinate to screen pixel integers
            screen_x = int(pos_matrix[i, 0] * self.scale)
            screen_y = int(pos_matrix[i, 1] * self.scale)

            # Draw the physical body of the agent as a small circle
            pygame.draw.circle(self.screen, self.COLOR_AGENT, (screen_x, screen_y), radius=BALL_SIZE)

            # Calculate heading vector endpoint to visualize orientation
            # We scale the velocity components so the heading indicator line is clearly visible
            vector_length_scale = ARROW_SIZE
            vx_component = vel_matrix[i, 0] * vector_length_scale
            vy_component = vel_matrix[i, 1] * vector_length_scale

            end_x = int(screen_x + vx_component)
            end_y = int(screen_y + vy_component)

            # Draw direction vector line
            pygame.draw.line(self.screen, self.COLOR_HEADING, (screen_x, screen_y), (end_x, end_y), width=2)

        # Swap buffers to render the current frame onto the monitor
        pygame.display.flip()

    def run(self, fps: int = 60, dt: float = 0.1) -> None:
        """
        The main visualization runtime loop. Controls clock ticks,
        physics updates, and graphic rendering.
        """
        running = True
        while running:
            # Handle user input / close events
            running = self.handle_events()
            if not running:
                break

            # Physics step: advance the simulation matrices
            # self.engine.step(dt)
            self.engine.step_with_alignment(dt)

            # Graphics step: draw the updated state
            self.render()

            # Throttle the loop to maintain steady frame rates
            self.clock.tick(fps)

        sys.exit()