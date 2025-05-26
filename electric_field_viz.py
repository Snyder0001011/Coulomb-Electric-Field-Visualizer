import numpy as np
import pygame
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import colorsys

# Constants
EPSILON_0 = 8.854187817e-12
K = 1 / (4 * math.pi * EPSILON_0)

# Color definitions
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (50, 50, 50)
DARK_GRAY = (30, 30, 30)
CYAN = (0, 255, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GRID_COLOR = (40, 40, 40)

@dataclass
class Charge:
    x: float
    y: float
    magnitude: float
    
    def get_color(self, selected=False):
        """Get color based on charge magnitude with smooth gradients"""
        if selected:
            return GREEN
        
        if self.magnitude > 0:
            # Positive charges: Blue to White gradient
            intensity = min(abs(self.magnitude) / 10, 1.0)
            b = 255
            r = int(100 + 155 * intensity)
            g = int(100 + 155 * intensity)
            return (r, g, b)
        else:
            # Negative charges: Red to Yellow gradient
            intensity = min(abs(self.magnitude) / 10, 1.0)
            r = 255
            g = int(50 + 205 * intensity)
            b = int(50 * (1 - intensity))
            return (r, g, b)

class ElectricFieldVisualizer:
    def __init__(self, width=1920, height=1080):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Enhanced Electric Field Visualizer")
        
        # Visualization parameters
        self.scale = 120
        self.arrow_spacing = 30
        self.show_grid = True
        self.show_field_lines = False
        self.show_potential = False
        self.show_equilibrium = True
        
        # Charges
        self.charges: List[Charge] = [
            Charge(1, -1, 1),
            Charge(-1, 1, -5),
        ]
        
        # Interaction state
        self.dragged_charge: Optional[int] = None
        self.selected_charge: Optional[int] = None
        self.pan_offset = [0, 0]
        self.is_panning = False
        self.last_mouse_pos = None
        
        # Performance
        self.equilibrium_points = []
        self.field_lines_cache = []
        self.potential_surface = None
        self.needs_update = True
        
        # UI
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
    def to_screen(self, x, y):
        """Convert world coordinates to screen coordinates"""
        return (
            int(self.width // 2 + x * self.scale + self.pan_offset[0]),
            int(self.height // 2 - y * self.scale + self.pan_offset[1])
        )
    
    def to_world(self, x, y):
        """Convert screen coordinates to world coordinates"""
        return (
            (x - self.width // 2 - self.pan_offset[0]) / self.scale,
            (self.height // 2 - y + self.pan_offset[1]) / self.scale
        )
    
    def calculate_field(self, px, py):
        """Calculate electric field at a point"""
        ex, ey = 0, 0
        for charge in self.charges:
            dx = px - charge.x
            dy = py - charge.y
            r_squared = dx**2 + dy**2
            
            if r_squared < 0.01:  # Avoid singularity
                continue
                
            r = np.sqrt(r_squared)
            ex += (K * charge.magnitude * dx) / r**3
            ey += (K * charge.magnitude * dy) / r**3
            
        return ex, ey
    
    def calculate_potential(self, px, py):
        """Calculate electric potential at a point"""
        v = 0
        for charge in self.charges:
            dx = px - charge.x
            dy = py - charge.y
            r = np.sqrt(dx**2 + dy**2)
            
            if r < 0.01:  # Avoid singularity
                continue
                
            v += K * charge.magnitude / r
            
        return v
    
    def find_equilibrium_points(self, threshold=None, spacing=0.05):
        """Find points where electric field is near zero"""
        if threshold is None:
            max_q = max(abs(c.magnitude) for c in self.charges)
            threshold = 0.0012 * K * max_q / 0.25
        
        points = []
        x_range = np.arange(-self.width/(2*self.scale), self.width/(2*self.scale), spacing)
        y_range = np.arange(-self.height/(2*self.scale), self.height/(2*self.scale), spacing)
        
        for x in x_range:
            for y in y_range:
                ex, ey = self.calculate_field(x, y)
                magnitude = np.sqrt(ex**2 + ey**2)
                
                if magnitude < threshold:
                    # Check if it's a true equilibrium (not just near a charge)
                    near_charge = any(
                        (x - c.x)**2 + (y - c.y)**2 < 0.1 
                        for c in self.charges
                    )
                    if not near_charge:
                        points.append((x, y))
        
        return points
    
    def calculate_field_lines(self, num_lines_per_charge=8):
        """Calculate field lines starting from charges"""
        lines = []
        
        for charge in self.charges:
            for i in range(num_lines_per_charge):
                angle = 2 * math.pi * i / num_lines_per_charge
                
                # Start slightly away from charge
                start_x = charge.x + 0.1 * math.cos(angle)
                start_y = charge.y + 0.1 * math.sin(angle)
                
                line = [(start_x, start_y)]
                x, y = start_x, start_y
                
                # Trace field line
                step_size = 0.05
                max_steps = 500
                
                for _ in range(max_steps):
                    ex, ey = self.calculate_field(x, y)
                    magnitude = np.sqrt(ex**2 + ey**2)
                    
                    if magnitude < 1e-6 or magnitude > 1e12:
                        break
                    
                    # Normalize and step
                    direction = 1 if charge.magnitude > 0 else -1
                    dx = direction * ex / magnitude * step_size
                    dy = direction * ey / magnitude * step_size
                    
                    x += dx
                    y += dy
                    
                    # Check boundaries
                    sx, sy = self.to_screen(x, y)
                    if sx < -100 or sx > self.width + 100 or sy < -100 or sy > self.height + 100:
                        break
                    
                    # Check if too close to another charge
                    if any((x - c.x)**2 + (y - c.y)**2 < 0.05 for c in self.charges):
                        break
                    
                    line.append((x, y))
                
                if len(line) > 5:
                    lines.append(line)
        
        return lines
    
    def calculate_potential_surface(self):
        """Calculate potential field for visualization"""
        resolution = 80
        x_range = np.linspace(-self.width/(2*self.scale), self.width/(2*self.scale), resolution)
        y_range = np.linspace(-self.height/(2*self.scale), self.height/(2*self.scale), resolution)
        
        surface = np.zeros((resolution, resolution))
        
        for i, y in enumerate(y_range):
            for j, x in enumerate(x_range):
                v = self.calculate_potential(x, y)
                surface[i, j] = v
        
        # Normalize for visualization
        v_max = np.percentile(np.abs(surface), 95)
        surface = np.clip(surface / v_max, -1, 1)
        
        return surface, x_range, y_range
    
    def draw_grid(self):
        """Draw coordinate grid"""
        # Grid lines
        grid_spacing = 100
        
        for x in range(0, self.width, grid_spacing):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, self.height), 1)
        
        for y in range(0, self.height, grid_spacing):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (self.width, y), 1)
        
        # Axes
        center_x = self.width // 2 + self.pan_offset[0]
        center_y = self.height // 2 + self.pan_offset[1]
        
        pygame.draw.line(self.screen, GRAY, (center_x, 0), (center_x, self.height), 2)
        pygame.draw.line(self.screen, GRAY, (0, center_y), (self.width, center_y), 2)
    
    def draw_field_arrows(self):
        """Draw electric field vectors"""
        for x in range(0, self.width, self.arrow_spacing):
            for y in range(0, self.height, self.arrow_spacing):
                wx, wy = self.to_world(x, y)
                ex, ey = self.calculate_field(wx, wy)
                
                magnitude = np.sqrt(ex**2 + ey**2)
                if magnitude < 1e-10 or magnitude > 1e12:
                    continue
                
                # Scale arrow length
                max_length = self.arrow_spacing * 0.8
                length = min(max_length, max_length * np.log10(magnitude + 1) / 10)
                
                fx = ex / magnitude * length
                fy = ey / magnitude * length
                
                start = (x, y)
                end = (x + fx, y - fy)
                
                # Color based on field strength
                intensity = min(1.0, np.log10(magnitude + 1) / 10)
                color = (
                    int(255 * intensity),
                    int(255 * (1 - intensity * 0.5)),
                    int(255 * (1 - intensity))
                )
                
                self.draw_arrow(color, start, end, arrow_size=5)
    
    def draw_arrow(self, color, start, end, arrow_size=5):
        """Draw an arrow with proper arrowhead"""
        pygame.draw.line(self.screen, color, start, end, 2)
        
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        if dx != 0 or dy != 0:
            angle = math.atan2(dy, dx)
            
            left = (
                end[0] - arrow_size * math.cos(angle - math.pi / 6),
                end[1] - arrow_size * math.sin(angle - math.pi / 6)
            )
            right = (
                end[0] - arrow_size * math.cos(angle + math.pi / 6),
                end[1] - arrow_size * math.sin(angle + math.pi / 6)
            )
            
            pygame.draw.polygon(self.screen, color, [end, left, right])
    
    def draw_field_lines(self):
        """Draw electric field lines"""
        for line in self.field_lines_cache:
            if len(line) < 2:
                continue
            
            points = [self.to_screen(x, y) for x, y in line]
            
            # Draw with gradient
            for i in range(len(points) - 1):
                t = i / len(points)
                color = (
                    int(255 * (1 - t * 0.5)),
                    int(255 * (1 - t * 0.3)),
                    255
                )
                pygame.draw.line(self.screen, color, points[i], points[i + 1], 2)
    
    def draw_potential_surface(self):
        """Draw equipotential contours"""
        if self.potential_surface is None:
            return
        
        surface, x_range, y_range = self.potential_surface
        
        # Draw contour lines
        contour_levels = np.linspace(-0.8, 0.8, 12)
        
        for level in contour_levels:
            points = []
            
            # Simple contour detection
            for i in range(len(y_range) - 1):
                for j in range(len(x_range) - 1):
                    val = surface[i, j]
                    
                    if (surface[i, j] < level <= surface[i, j + 1] or
                        surface[i, j] >= level > surface[i, j + 1]):
                        
                        t = (level - surface[i, j]) / (surface[i, j + 1] - surface[i, j])
                        x = x_range[j] + t * (x_range[j + 1] - x_range[j])
                        y = y_range[i]
                        points.append(self.to_screen(x, y))
            
            if len(points) > 2:
                # Color based on potential
                if level > 0:
                    color = (100, 100, 255)
                elif level < 0:
                    color = (255, 100, 100)
                else:
                    color = (200, 200, 200)
                
                for point in points:
                    pygame.draw.circle(self.screen, color, point, 1)
    
    def draw_charges(self):
        """Draw all charges with enhanced visuals"""
        for i, charge in enumerate(self.charges):
            pos = self.to_screen(charge.x, charge.y)
            
            # Draw charge
            radius = int(20 + min(10, abs(charge.magnitude)))
            color = charge.get_color(selected=(i == self.selected_charge))
            
            # Glow effect
            for r in range(radius + 10, radius, -2):
                alpha = (radius + 10 - r) / 10
                glow_color = tuple(int(c * alpha) for c in color)
                pygame.draw.circle(self.screen, glow_color, pos, r, 1)
            
            pygame.draw.circle(self.screen, color, pos, radius)
            
            # Draw charge symbol
            symbol_color = BLACK if sum(color) > 400 else WHITE
            if charge.magnitude > 0:
                # Plus sign
                pygame.draw.line(self.screen, symbol_color, 
                               (pos[0] - radius//2, pos[1]), 
                               (pos[0] + radius//2, pos[1]), 3)
                pygame.draw.line(self.screen, symbol_color, 
                               (pos[0], pos[1] - radius//2), 
                               (pos[0], pos[1] + radius//2), 3)
            else:
                # Minus sign
                pygame.draw.line(self.screen, symbol_color, 
                               (pos[0] - radius//2, pos[1]), 
                               (pos[0] + radius//2, pos[1]), 3)
            
            # Draw magnitude label
            label = self.small_font.render(f"{charge.magnitude:.1f}", True, WHITE)
            self.screen.blit(label, (pos[0] + radius + 5, pos[1] - 10))
            
            # Selection indicator
            if i == self.dragged_charge:
                pygame.draw.circle(self.screen, GREEN, pos, radius + 5, 2)
    
    def draw_equilibrium_points(self):
        """Draw equilibrium points"""
        for x, y in self.equilibrium_points:
            pos = self.to_screen(x, y)
            
            # Draw with pulsing effect
            time = pygame.time.get_ticks() / 1000
            size = int(6 + 2 * math.sin(time * 3))
            
            pygame.draw.circle(self.screen, CYAN, pos, size)
            pygame.draw.circle(self.screen, WHITE, pos, size, 1)
    
    def draw_ui(self):
        """Draw user interface elements"""
        # Help text
        help_lines = [
            "Controls:",
            "Click and drag - Move charges",
            "Right click drag - Pan view",
            "Scroll - Zoom in/out",
            "A - Add positive charge",
            "S - Add negative charge",
            "D - Delete selected charge",
            "G - Toggle grid",
            "F - Toggle field lines",
            "P - Toggle potential",
            "E - Toggle equilibrium points",
            "R - Reset view",
            "ESC - Exit"
        ]
        
        y = 10
        for line in help_lines:
            text = self.small_font.render(line, True, WHITE)
            self.screen.blit(text, (10, y))
            y += 20
        
        # Status
        status_text = f"Charges: {len(self.charges)} | "
        if self.show_field_lines:
            status_text += "Field Lines: ON | "
        if self.show_potential:
            status_text += "Potential: ON | "
        
        status = self.font.render(status_text, True, WHITE)
        self.screen.blit(status, (10, self.height - 30))
    
    def handle_event(self, event):
        """Handle pygame events"""
        if event.type == pygame.QUIT:
            return False
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            
            if event.button == 1:  # Left click
                # Check if clicking on a charge
                for i, charge in enumerate(self.charges):
                    sx, sy = self.to_screen(charge.x, charge.y)
                    if (mx - sx)**2 + (my - sy)**2 < 25**2:
                        self.dragged_charge = i
                        self.selected_charge = i
                        break
            
            elif event.button == 3:  # Right click - pan
                self.is_panning = True
                self.last_mouse_pos = event.pos
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragged_charge = None
            elif event.button == 3:
                self.is_panning = False
        
        elif event.type == pygame.MOUSEMOTION:
            if self.dragged_charge is not None:
                wx, wy = self.to_world(*event.pos)
                self.charges[self.dragged_charge].x = wx
                self.charges[self.dragged_charge].y = wy
                self.needs_update = True
            
            elif self.is_panning and self.last_mouse_pos:
                dx = event.pos[0] - self.last_mouse_pos[0]
                dy = event.pos[1] - self.last_mouse_pos[1]
                self.pan_offset[0] += dx
                self.pan_offset[1] += dy
                self.last_mouse_pos = event.pos
                self.needs_update = True
        
        elif event.type == pygame.MOUSEWHEEL:
            # Zoom
            zoom_factor = 1.1 if event.y > 0 else 0.9
            self.scale *= zoom_factor
            self.scale = max(20, min(500, self.scale))
            self.needs_update = True
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            
            elif event.key == pygame.K_a:
                # Add positive charge at center
                wx, wy = self.to_world(self.width // 2, self.height // 2)
                self.charges.append(Charge(wx, wy, 1))
                self.needs_update = True
            
            elif event.key == pygame.K_s:
                # Add negative charge at center
                wx, wy = self.to_world(self.width // 2, self.height // 2)
                self.charges.append(Charge(wx, wy, -1))
                self.needs_update = True
            
            elif event.key == pygame.K_d and self.selected_charge is not None:
                # Delete selected charge
                if len(self.charges) > 1:
                    del self.charges[self.selected_charge]
                    self.selected_charge = None
                    self.needs_update = True
            
            elif event.key == pygame.K_g:
                self.show_grid = not self.show_grid
            
            elif event.key == pygame.K_f:
                self.show_field_lines = not self.show_field_lines
                self.needs_update = True
            
            elif event.key == pygame.K_p:
                self.show_potential = not self.show_potential
                self.needs_update = True
            
            elif event.key == pygame.K_e:
                self.show_equilibrium = not self.show_equilibrium
            
            elif event.key == pygame.K_r:
                # Reset view
                self.pan_offset = [0, 0]
                self.scale = 120
                self.needs_update = True
        
        return True
    
    def update(self):
        """Update cached calculations if needed"""
        if self.needs_update:
            if self.show_equilibrium:
                self.equilibrium_points = self.find_equilibrium_points()
            
            if self.show_field_lines:
                self.field_lines_cache = self.calculate_field_lines()
            
            if self.show_potential:
                self.potential_surface = self.calculate_potential_surface()
            
            self.needs_update = False
    
    def draw(self):
        """Draw everything"""
        self.screen.fill(BLACK)
        
        if self.show_grid:
            self.draw_grid()
        
        if self.show_potential:
            self.draw_potential_surface()
        
        self.draw_field_arrows()
        
        if self.show_field_lines:
            self.draw_field_lines()
        
        if self.show_equilibrium:
            self.draw_equilibrium_points()
        
        self.draw_charges()
        self.draw_ui()
        
        pygame.display.flip()
    
    def run(self):
        """Main loop"""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if not self.handle_event(event):
                    running = False
            
            self.update()
            self.draw()
            clock.tick(60)
        
        pygame.quit()

# Run the visualizer
if __name__ == "__main__":
    visualizer = ElectricFieldVisualizer()
    visualizer.run()