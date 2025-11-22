import pygame
import numpy as np
import sys
import time

# Constants
GAME_SIZE = 64 # Each mini-game is 64x64 pixels
FPS = 60

# Color Modes
MODE_CLASSIC = 0
MODE_AGE = 1
MODE_TRAILS = 2
MODE_ACTIVITY = 3
NUM_MODES = 4

class MassiveLife:
    def __init__(self):
        pygame.init()
        # Fullscreen mode
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.width, self.height = self.screen.get_size()
        
        pygame.display.set_caption("Massive Parallel Game of Life (NumPy Optimized)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 24)
        
        # Calculate grid dimensions based on screen size
        self.rows = self.height // GAME_SIZE
        self.cols = self.width // GAME_SIZE
        self.num_games = self.rows * self.cols
        
        # Global grids
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)
        self.age_grid = np.zeros((self.height, self.width), dtype=np.int16)
        self.fade_grid = np.zeros((self.height, self.width), dtype=np.float32)
        self.activity_grid = np.zeros((self.height, self.width), dtype=np.float32)
        
        self.current_mode = MODE_CLASSIC
        self.mode_names = ["Classic", "Age (Blue=Old)", "Trails (Red=History)", "Activity (Heatmap)"]
        
        # Track metadata for each game to handle resets
        self.game_states = []
        for r in range(self.rows):
            for c in range(self.cols):
                self.game_states.append({
                    'row': r,
                    'col': c,
                    'y_start': r * GAME_SIZE,
                    'x_start': c * GAME_SIZE,
                    'generation': 0,
                    'max_gen': np.random.randint(200, 800),
                    'stable_frames': 0,
                    'prev_grid': None
                })
                self.reset_game(self.game_states[-1])

    def reset_game(self, game_state):
        y, x = game_state['y_start'], game_state['x_start']
        density = np.random.uniform(0.1, 0.6)
        
        # Reset main grid
        self.grid[y:y+GAME_SIZE, x:x+GAME_SIZE] = np.random.choice(
            [0, 1], 
            size=(GAME_SIZE, GAME_SIZE), 
            p=[1-density, density]
        )
        
        # Reset aux grids
        self.age_grid[y:y+GAME_SIZE, x:x+GAME_SIZE] = 0
        self.fade_grid[y:y+GAME_SIZE, x:x+GAME_SIZE] = 0
        self.activity_grid[y:y+GAME_SIZE, x:x+GAME_SIZE] = 0
        
        game_state['generation'] = 0
        game_state['stable_frames'] = 0
        game_state['prev_grid'] = np.zeros((GAME_SIZE, GAME_SIZE), dtype=np.int8)

    def update(self):
        # Vectorized neighbor counting
        N = (np.roll(self.grid, 1, axis=0) + np.roll(self.grid, -1, axis=0) +
             np.roll(self.grid, 1, axis=1) + np.roll(self.grid, -1, axis=1) +
             np.roll(np.roll(self.grid, 1, axis=0), 1, axis=1) +
             np.roll(np.roll(self.grid, 1, axis=0), -1, axis=1) +
             np.roll(np.roll(self.grid, -1, axis=0), 1, axis=1) +
             np.roll(np.roll(self.grid, -1, axis=0), -1, axis=1))

        # Apply rules
        birth = (N == 3) & (self.grid == 0)
        survive = ((N == 2) | (N == 3)) & (self.grid == 1)
        
        new_grid = np.zeros_like(self.grid)
        new_grid[birth | survive] = 1
        
        # Update Aux Grids
        # Age: Increment if alive, reset if dead (or born)
        # Actually, if it survives, increment. If born, set to 1.
        self.age_grid = np.where(new_grid == 1, self.age_grid + 1, 0)
        
        # Fade: If dead, decay. If alive, set to 1.0
        self.fade_grid = np.where(new_grid == 1, 1.0, self.fade_grid * 0.9)
        
        # Activity: Decay existing, add 1.0 if state changed
        changed = (self.grid != new_grid)
        self.activity_grid = self.activity_grid * 0.95
        self.activity_grid[changed] = 1.0
        
        # Check for resets
        for game in self.game_states:
            y, x = game['y_start'], game['x_start']
            sub_grid = new_grid[y:y+GAME_SIZE, x:x+GAME_SIZE]
            
            # Check stability
            if np.array_equal(sub_grid, game['prev_grid']):
                game['stable_frames'] += 1
            else:
                game['stable_frames'] = 0
            
            game['prev_grid'] = sub_grid.copy()
            game['generation'] += 1
            
            # Reset condition
            if (game['stable_frames'] > 10 or 
                game['generation'] > game['max_gen'] or 
                np.sum(sub_grid) == 0):
                
                density = np.random.uniform(0.1, 0.6)
                new_grid[y:y+GAME_SIZE, x:x+GAME_SIZE] = np.random.choice(
                    [0, 1], 
                    size=(GAME_SIZE, GAME_SIZE), 
                    p=[1-density, density]
                )
                # Reset aux grids for this region
                self.age_grid[y:y+GAME_SIZE, x:x+GAME_SIZE] = 0
                self.fade_grid[y:y+GAME_SIZE, x:x+GAME_SIZE] = 0
                self.activity_grid[y:y+GAME_SIZE, x:x+GAME_SIZE] = 0
                
                game['generation'] = 0
                game['stable_frames'] = 0
                game['prev_grid'] = np.zeros((GAME_SIZE, GAME_SIZE), dtype=np.int8)

        self.grid = new_grid

    def draw(self):
        rgb_array = np.zeros((self.width, self.height, 3), dtype=np.uint8)
        grid_t = self.grid.T
        
        if self.current_mode == MODE_CLASSIC:
            rgb_array[grid_t == 1] = [255, 255, 255]
            
        elif self.current_mode == MODE_AGE:
            # White for new, fading to Blue/Purple for old
            age_t = self.age_grid.T
            mask = (grid_t == 1)
            
            # Normalize age for color (cap at 100)
            norm_age = np.clip(age_t, 0, 100) / 100.0
            
            # R: 255 -> 50
            # G: 255 -> 50
            # B: 255 -> 255
            r = (255 * (1 - norm_age) + 50 * norm_age).astype(np.uint8)
            g = (255 * (1 - norm_age) + 50 * norm_age).astype(np.uint8)
            b = np.full_like(r, 255)
            
            rgb_array[mask, 0] = r[mask]
            rgb_array[mask, 1] = g[mask]
            rgb_array[mask, 2] = b[mask]
            
        elif self.current_mode == MODE_TRAILS:
            # White for alive, Red fade for dead
            fade_t = self.fade_grid.T
            
            # Alive cells are white
            rgb_array[grid_t == 1] = [255, 255, 255]
            
            # Dead cells with fade > 0.1 are red
            mask_trail = (grid_t == 0) & (fade_t > 0.05)
            val = (fade_t[mask_trail] * 255).astype(np.uint8)
            
            rgb_array[mask_trail, 0] = val # Red channel
            
        elif self.current_mode == MODE_ACTIVITY:
            # Heatmap: Black -> Red -> Yellow -> White
            act_t = self.activity_grid.T
            
            # Simple gradient: Red channel always high for active, Green increases with intensity
            mask = (act_t > 0.01)
            val = (act_t[mask] * 255).astype(np.uint8)
            
            rgb_array[mask, 0] = val # Red
            rgb_array[mask, 1] = (val * 0.5).astype(np.uint8) # Green (Orange-ish)
            # rgb_array[mask, 2] = 0 # Blue
            
            # Overlay alive cells as white for clarity? Or just let the heatmap speak?
            # Let's overlay white for current alive cells to see structure
            rgb_array[grid_t == 1] = [255, 255, 255]

        pygame.surfarray.blit_array(self.screen, rgb_array)
        
        # Draw UI Overlay
        mode_text = self.font.render(f"Mode: {self.mode_names[self.current_mode]} (TAB to switch)", True, (0, 255, 0))
        self.screen.blit(mode_text, (20, 20))
        
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_TAB:
                        self.current_mode = (self.current_mode + 1) % NUM_MODES

            self.update()
            self.draw()
            self.clock.tick(FPS)
            
            # Show FPS in caption (though fullscreen hides it, good for debugging if windowed)
            # pygame.display.set_caption(f"Massive Parallel Life - FPS: {self.clock.get_fps():.2f}")

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    MassiveLife().run()
