import pygame
import numpy as np
import sys
import random
from god import EvolutionStrategy, NeuralNetwork, GRID_WIDTH, GRID_HEIGHT, POPULATION_SIZE

# Constants
GAME_SIZE = 32
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60
MAX_AGE = 600 # Max frames before forced death (10 seconds)
STAGNATION_THRESHOLD = 20 # Increased to 20 to kill more complex oscillators
STAGNATION_LIMIT = 20 # Reduced to 20 frames (0.3s) for faster cleanup


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (50, 255, 50)

class MassiveGodMode:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Massive God Mode: ES Evolution")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 24)
        
        self.es = EvolutionStrategy()
        
        # Calculate layout
        self.cols = SCREEN_WIDTH // GAME_SIZE
        self.rows = SCREEN_HEIGHT // GAME_SIZE
        self.max_games = self.cols * self.rows
        
        # Adjust screen size
        self.sim_width = self.cols * GAME_SIZE
        self.sim_height = self.rows * GAME_SIZE
        self.screen = pygame.display.set_mode((self.sim_width, self.sim_height))
        
        # Global grid
        self.grid = np.zeros((self.sim_height, self.sim_width), dtype=np.int8)
        
        # Stats Arrays (Vectorized tracking)
        # Shape: (rows, cols)
        self.stats_age = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.stats_stagnation = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.stats_initial_pop = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.stats_activity_accum = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.stats_alive = np.ones((self.rows, self.cols), dtype=bool)
        
        # Population Management
        # We keep a "Hall of Fame" of genomes to breed from
        self.hall_of_fame = []
        self.best_fitness_ever = 0
        
        # Active Genomes Grid (to track which network is where)
        self.active_genomes = [[NeuralNetwork() for _ in range(self.cols)] for _ in range(self.rows)]
        
        # Isolation Mask (Void Barriers)
        # 1 = Alive allowed, 0 = Dead zone
        self.isolation_mask = np.ones((self.sim_height, self.sim_width), dtype=np.int8)
        
        # Create dead zones at borders
        # We want a 1px border around each GAME_SIZE block
        # Actually, just the edges of each block.
        # Vertical lines
        for c in range(self.cols + 1):
            x = c * GAME_SIZE
            if x < self.sim_width:
                self.isolation_mask[:, x] = 0
            if x - 1 >= 0:
                self.isolation_mask[:, x-1] = 0
                
        # Horizontal lines
        for r in range(self.rows + 1):
            y = r * GAME_SIZE
            if y < self.sim_height:
                self.isolation_mask[y, :] = 0
            if y - 1 >= 0:
                self.isolation_mask[y-1, :] = 0
                
        # Pre-render Grid Overlay
        self.grid_overlay = pygame.Surface((self.sim_width, self.sim_height), pygame.SRCALPHA)
        grid_color = (30, 30, 30) # Faint grey
        
        # Vertical lines
        for c in range(1, self.cols):
            x = c * GAME_SIZE
            pygame.draw.line(self.grid_overlay, grid_color, (x, 0), (x, self.sim_height))
            
        # Horizontal lines
        for r in range(1, self.rows):
            y = r * GAME_SIZE
            pygame.draw.line(self.grid_overlay, grid_color, (0, y), (self.sim_width, y))
        
        # Initialize all games
        print("Initializing population...")
        for r in range(self.rows):
            for c in range(self.cols):
                self.reset_game(r, c, initial=True)

    def reset_game(self, r, c, initial=False, target_grid=None):
        # Create new child
        child = NeuralNetwork()
        
        if not initial and len(self.hall_of_fame) > 2:
            # Breed from Hall of Fame
            # Tournament selection
            parents = random.sample(self.hall_of_fame, min(len(self.hall_of_fame), 5))
            parents.sort(key=lambda x: x[0], reverse=True)
            parent_genome = parents[0][1] # Best of sample
            
            child.set_genome(parent_genome)
            child.mutate()
        elif not initial:
             # Random mutation of current best if HoF is small
             if self.es.best_genome is not None:
                 child.set_genome(self.es.best_genome)
                 child.mutate()
        
        self.active_genomes[r][c] = child
        
        # Generate Seed  
        seed = self.es.generate_grid(child)
        
        # Place in grid
        y = r * GAME_SIZE
        x = c * GAME_SIZE
        
        # Determine which grid to write to
        if target_grid is None:
            target_grid = self.grid

        # Apply mask to seed to ensure it doesn't start in void
        # (Though seed is 32x32 and we might have masked edges)
        # Let's just place it and let the mask kill edges next frame
        target_grid[y:y+GAME_SIZE, x:x+GAME_SIZE] = seed
        
        # Reset Stats
        self.stats_age[r, c] = 0
        self.stats_stagnation[r, c] = 0
        self.stats_initial_pop[r, c] = np.sum(seed)
        self.stats_activity_accum[r, c] = 0
        self.stats_alive[r, c] = True

    def update(self):
        # 1. Global Game of Life Update (NumPy Optimized)
        current_grid = self.grid
        N = (np.roll(current_grid, 1, axis=0) + np.roll(current_grid, -1, axis=0) +
             np.roll(current_grid, 1, axis=1) + np.roll(current_grid, -1, axis=1) +
             np.roll(np.roll(current_grid, 1, axis=0), 1, axis=1) +
             np.roll(np.roll(current_grid, 1, axis=0), -1, axis=1) +
             np.roll(np.roll(current_grid, -1, axis=0), 1, axis=1) +
             np.roll(np.roll(current_grid, -1, axis=0), -1, axis=1))
        
        birth = (N == 3) & (current_grid == 0)
        survive = ((N == 2) | (N == 3)) & (current_grid == 1)
        new_grid = np.zeros_like(current_grid)
        new_grid[birth | survive] = 1
        
        # Apply Void Barriers (Bitwise AND is faster than multiplication for 0/1)
        new_grid &= self.isolation_mask
        
        # 2. Calculate Metrics per Game (Vectorized)
        # Reshape to (rows, GAME_SIZE, cols, GAME_SIZE) to sum per block
        # We want to sum over axis 1 and 3 (the internal block dimensions)
        
        # Activity
        changed = (current_grid != new_grid).astype(np.int32)
        reshaped_changed = changed.reshape(self.rows, GAME_SIZE, self.cols, GAME_SIZE)
        activity_per_game = reshaped_changed.sum(axis=(1, 3))
        
        # Population
        reshaped_new = new_grid.reshape(self.rows, GAME_SIZE, self.cols, GAME_SIZE)
        pop_per_game = reshaped_new.sum(axis=(1, 3))
        
        # 3. Update Stats Arrays
        self.stats_age += 1
        self.stats_activity_accum += activity_per_game
        
        # Stagnation Check
        # If activity is low, increment stagnation counter. Else reset it.
        is_stagnant = activity_per_game < STAGNATION_THRESHOLD
        self.stats_stagnation[is_stagnant] += 1
        self.stats_stagnation[~is_stagnant] = 0
        
        # 4. Identify Dead/Stagnant Games
        # Conditions for reset:
        # - Population is 0 (Dead)
        # - Stagnation > Limit (Boring)
        # - Age > Max (Old age death to make room)
        
        dead_mask = (pop_per_game == 0)
        stagnant_mask = (self.stats_stagnation > STAGNATION_LIMIT)
        old_mask = (self.stats_age > MAX_AGE)
        
        reset_mask = dead_mask | stagnant_mask | old_mask
        
        # 5. Process Resets (Python Loop but only for resetting games)
        # Get indices of games to reset
        rows_to_reset, cols_to_reset = np.where(reset_mask)
        
        # Debug Backlog
        if hasattr(self, 'frame_count') and self.frame_count % 60 == 0 and len(rows_to_reset) > 0:
            print(f"Reset Queue: {len(rows_to_reset)} games waiting.")
        elif not hasattr(self, 'frame_count'):
             self.frame_count = 0
        self.frame_count += 1

        # Limit resets per frame to prevent lag spikes
        MAX_RESETS = 100 # Increased to 100 to clear backlog instantly
        if len(rows_to_reset) > MAX_RESETS:
            # Randomly select subset to reset
            indices = np.random.choice(len(rows_to_reset), MAX_RESETS, replace=False)
            rows_to_reset = rows_to_reset[indices]
            cols_to_reset = cols_to_reset[indices]
        
        if len(rows_to_reset) > 0:
            # Calculate Spread (Bounding Box) only for finishing games
            # This is expensive so we only do it here
            
            for i in range(len(rows_to_reset)):
                r, c = rows_to_reset[i], cols_to_reset[i]
                
                # Calculate Fitness
                final_pop = pop_per_game[r, c]
                initial_pop = self.stats_initial_pop[r, c]
                duration = self.stats_age[r, c]
                total_activity = self.stats_activity_accum[r, c]
                
                # Spread Calculation
                # Extract grid for this game
                y, x = r * GAME_SIZE, c * GAME_SIZE
                game_grid = new_grid[y:y+GAME_SIZE, x:x+GAME_SIZE]
                
                spread = 0
                if final_pop > 0:
                    rows, cols = np.where(game_grid == 1)
                    if len(rows) > 0:
                        h = np.max(rows) - np.min(rows) + 1
                        w = np.max(cols) - np.min(cols) + 1
                        spread = (h * w) / (GAME_SIZE * GAME_SIZE)
                
                # Fitness Formula
                score = duration *1.0
                score += total_activity * 0.5
                if initial_pop > 0:
                    score += (final_pop / initial_pop) * 10.0
                score += spread * 50.0
                
                # Penalties
                if initial_pop == 0: score -= 100
                elif total_activity < 5: score -= 50
                
                score = max(score, 0)
                
                # Update Hall of Fame
                if score > 100: # Min threshold to enter
                    genome = self.active_genomes[r][c].get_genome()
                    self.hall_of_fame.append((score, genome))
                    # Keep HoF size manageable
                    if len(self.hall_of_fame) > 100:
                        self.hall_of_fame.sort(key=lambda x: x[0], reverse=True)
                        self.hall_of_fame = self.hall_of_fame[:50] # Keep top 50
                
                if score > self.best_fitness_ever:
                    self.best_fitness_ever = score
                
                # Reset the game
                self.reset_game(r, c, target_grid=new_grid)
        
        # Apply grid update
        self.grid = new_grid

    def draw(self):
        # Render
        rgb_array = np.zeros((self.sim_width, self.sim_height, 3), dtype=np.uint8)
        grid_t = self.grid.T
        
        # White cells
        rgb_array[grid_t == 1] = [255, 255, 255]
        
        pygame.surfarray.blit_array(self.screen, rgb_array)
        
        # Draw Grid Overlay (Pre-rendered)
        self.screen.blit(self.grid_overlay, (0, 0))
        
        # Overlay
        text = self.font.render(f"Best Fitness: {self.best_fitness_ever:.0f} | HoF Size: {len(self.hall_of_fame)}", True, GREEN)
        self.screen.blit(text, (20, 20))
        
        pygame.display.flip()

    def run(self):
        # Try to load existing progress
        self.es.load()
        
        running = True
        frame_counter = 0
        
        try:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                
                self.update()
                self.draw()
                self.clock.tick(FPS)
                
                frame_counter += 1
                # Auto-save every 600 frames (10 seconds at 60 FPS)
                if frame_counter % 600 == 0:
                    self.es.save()
        except KeyboardInterrupt:
            print("\nSaving progress...")
            self.es.save()

        # Save on exit
        self.es.save()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    MassiveGodMode().run()
