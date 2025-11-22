import pygame
import numpy as np
import sys
from god import EvolutionStrategy, GRID_WIDTH, GRID_HEIGHT

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
CELL_SIZE = 10 # Visual size
SIM_WIDTH = GRID_WIDTH * CELL_SIZE
SIM_HEIGHT = GRID_HEIGHT * CELL_SIZE
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (50, 50, 50)
BLUE = (50, 50, 255)

class GodView:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("God Mode: ES Evolutionary Learning")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 16)
        
        self.es = EvolutionStrategy()
        self.best_grid = None
        self.sim_grid = None
        self.sim_timer = 0
        
        # Graph data
        self.graph_width = SCREEN_WIDTH - SIM_WIDTH - 40
        self.graph_height = 200
        self.graph_x = SIM_WIDTH + 20
        self.graph_y = 50

    def update_simulation(self):
        # If we have a best genome, simulate it visually
        if self.es.best_genome is not None:
            if self.sim_grid is None or self.sim_timer > 100: # Reset every 100 frames
                self.es.mean_network.set_genome(self.es.best_genome)
                self.sim_grid = self.es.generate_grid(self.es.mean_network)
                self.sim_timer = 0
            
            # Run one step of simulation
            current_grid = self.sim_grid
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
            
            self.sim_grid = new_grid
            self.sim_timer += 1

    def draw(self):
        self.screen.fill(BLACK)
        
        # Draw Simulation (Left)
        pygame.draw.rect(self.screen, GRAY, (0, 0, SIM_WIDTH, SIM_HEIGHT), 1)
        if self.sim_grid is not None:
            for y in range(GRID_HEIGHT):
                for x in range(GRID_WIDTH):
                    if self.sim_grid[y, x] == 1:
                        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                        pygame.draw.rect(self.screen, WHITE, rect)
        
        # Draw Stats (Right)
        stats = [
            f"Generation: {self.es.generation}",
            f"Best Fitness: {self.es.best_fitness:.2f}",
            f"Method: Evolution Strategies",
            "",
            "Training with ES...",
            "Visualizing Best Genome"
        ]
        
        y_off = 20
        for line in stats:
            text = self.font.render(line, True, GREEN)
            self.screen.blit(text, (self.graph_x, y_off))
            y_off += 25
            
        # Draw Fitness Graph
        pygame.draw.rect(self.screen, GRAY, (self.graph_x, self.graph_y + 150, self.graph_width, self.graph_height))
        if len(self.es.fitness_history) > 1:
            points = []
            max_fit = max([x['best_fitness'] for x in self.es.fitness_history]) if self.es.fitness_history else 1
            for i, entry in enumerate(self.es.fitness_history[-self.graph_width:]): # Last N points
                x = self.graph_x + i
                y = (self.graph_y + 150 + self.graph_height) - (entry['best_fitness'] / max_fit * self.graph_height)
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, BLUE, False, points, 2)
                
        pygame.display.flip()

    def run(self):
        running = True
        
        # Try to load existing progress
        self.es.load()
        
        try:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                
                # Evolve one step per frame (or every few frames if slow)
                self.es.evolve()
                
                # Auto-save every 10 generations
                if self.es.generation % 10 == 0 and self.es.generation > 0:
                    self.es.save()
                
                self.update_simulation()
                self.draw()
                self.clock.tick(FPS)
        except KeyboardInterrupt:
            print("\nSaving progress...")
            self.es.save()
            
        # Save on exit
        self.es.save()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    GodView().run()
