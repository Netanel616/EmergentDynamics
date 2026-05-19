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
        pygame.font.init()
        pygame.display.set_caption("EmergentDynamics - Multi-Agent Simulation")

        # Allocate screen surface
        self.screen: pygame.Surface = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock: pygame.time.Clock = pygame.time.Clock()
        # Initialize default vector font for HUD overlay (compiles everywhere without external assets)
        self.font = pygame.font.Font(None, 22)
        self.title_font = pygame.font.Font(None, 24)

        # Interactive UI/UX States
        self.use_smooth_mode: bool = True  # Starts in the realistic continuous smooth mode
        self.align_on: bool = True
        self.show_hud: bool = True  # Press 'H' to hide/show on-screen documentation

        # Visual color palette (Dark Slate, Vibrant Emerald, Amber vector headings)
        self.COLOR_BG = (28, 33, 41)  # Dark slate background
        self.COLOR_AGENT = (16, 185, 129)  # Vibrant Emerald green for agents
        self.COLOR_HEADING = (245, 158, 11)  # Amber vectors to show directions
        self.COLOR_TEXT_WHITE = (248, 250, 252)  # Light slate for clean text
        self.COLOR_TEXT_TEAL = (45, 212, 191)  # Bright teal for accent states

    def handle_events(self) -> bool:
        """
        Processes OS windows events.
        Returns False if the window is closed, indicating the simulation must stop.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return False

                # Dynamic Physics Toggle: Switch between Bumpy and Smooth loops
                elif event.key == pygame.K_SPACE:
                    self.use_smooth_mode = not self.use_smooth_mode

                # HUD Visibility Toggle
                elif event.key == pygame.K_h:
                    self.show_hud = not self.show_hud
                # Align Toggle
                elif event.key == pygame.K_c:
                    self.align_on = not self.align_on


                # Noise Amplitude Tweak (Increment of 0.05)
                elif event.key in (pygame.K_UP, pygame.K_EQUALS):
                    self.engine.set_noise_amplitude(self.engine.eta + 0.05)
                elif event.key in (pygame.K_DOWN, pygame.K_MINUS):
                    self.engine.set_noise_amplitude(self.engine.eta - 0.05)

                # Attack/Turn Rate Limit Tweak (Increment of 0.5 rad/s)
                elif event.key == pygame.K_w:
                    self.engine.max_turn_rate = min(15.0, self.engine.max_turn_rate + 0.5)
                elif event.key == pygame.K_s:
                    self.engine.max_turn_rate = max(0.5, self.engine.max_turn_rate - 0.5)

        return True

    def render_hud(self) -> None:
        """
        Renders a sleek, semi-transparent Heads-Up Display panel in the upper-left
        corner, showing real-time parameters, frame rates, and control bindings.
        """
        if not self.show_hud:
            return

        # 1. Allocate a semi-transparent panel surface (Slate 900 with alpha)
        hud_width, hud_height = 360, 190
        hud_surface = pygame.Surface((hud_width, hud_height), pygame.SRCALPHA)
        hud_surface.fill((15, 23, 42, 220))  # RGBA

        # 2. Add an elegant accent border
        pygame.draw.rect(hud_surface, self.COLOR_HEADING, (0, 0, hud_width, hud_height), width=1)

        # 3. Compile lines of state info
        fps_text = f"FPS: {int(self.clock.get_fps())}"
        align_text = "Align On" if self.align_on else "Align Off"
        mode_text = "SMOOTH (Rotational Inertia)" if self.use_smooth_mode else "CLASSIC (Instant Snapping)"
        noise_text = f"Noise (eta): {self.engine.eta:.2f} rad"
        turn_text = f"Turn Rate (omega): {self.engine.max_turn_rate:.1f} rad/s"
        stride_text = f"Neighbor Stride: {self.engine.update_stride} frame(s)"
        pop_text = f"Population (N): {self.engine.N} agents"

        # 4. Draw texts on the transparent HUD surface
        # Header Title
        title_render = self.title_font.render("EmergentDynamics Control HUD", True, self.COLOR_HEADING)
        hud_surface.blit(title_render, (15, 12))

        # Dynamic variable rows
        rows = [
            (pop_text, self.COLOR_TEXT_WHITE),
            (mode_text, self.COLOR_TEXT_TEAL),
            (align_text, self.COLOR_TEXT_TEAL),
            (noise_text, self.COLOR_TEXT_WHITE),
            (turn_text, self.COLOR_TEXT_WHITE),
            (fps_text, self.COLOR_TEXT_WHITE)
        ]

        for idx, (text_line, color) in enumerate(rows):
            line_surface = self.font.render(text_line, True, color)
            hud_surface.blit(line_surface, (15, 42 + (idx * 22)))

        # Guide controls along the footer
        guide_surface = self.font.render("SPACE: Toggle Mode | UP/DN: Noise | W/S: Turn Rate | H: HUD", True,
                                         (148, 163, 184))
        hud_surface.blit(guide_surface, (15, 162))

        # 5. Composite HUD onto the main display window
        self.screen.blit(hud_surface, (15, 15))

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

        self.render_hud()
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

            if self.align_on:
                # Physics step: advance the simulation matrices
                # self.engine.step(dt)
                if self.use_smooth_mode:
                    self.engine.step_with_smooth_alignment(dt)
                else:
                    self.engine.step_with_alignment(dt)
            else:
                self.engine.step(dt)


            # Graphics step: draw the updated state
            self.render()

            # Throttle the loop to maintain steady frame rates
            self.clock.tick(fps)

        sys.exit()