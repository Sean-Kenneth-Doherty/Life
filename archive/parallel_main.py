import pygame
import sys
import random
import json
import os
import time

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 900
ROWS = 4
COLS = 4
NUM_GAMES = ROWS * COLS
GAME_WIDTH = SCREEN_WIDTH // COLS
GAME_HEIGHT = SCREEN_HEIGHT // ROWS
CELL_SIZE = 4 # Smaller cells for mini-grids
GRID_WIDTH = GAME_WIDTH // CELL_SIZE
GRID_HEIGHT = GAME_HEIGHT // CELL_SIZE
FPS = 30

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

class MiniGame:
    def __init__(self, index):
        self.index = index
        self.row = index // COLS
        self.col = index % COLS
        self.x_offset = self.col * GAME_WIDTH
        self.y_offset = self.row * GAME_HEIGHT
        self.reset()

    def reset(self):
        self.density = random.uniform(0.1, 0.6)
        self.grid = [[1 if random.random() < self.density else 0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.initial_grid = [row[:] for row in self.grid]
        self.initial_pop = sum(sum(row) for row in self.grid)
        self.generation = 0
        self.population_history = []
        self.activity_history = []
        self.stable_frames = 0
        self.max_generations = random.randint(200, 600)

    def update(self):
        new_grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        population = 0
        activity = 0
        
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                count = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                            count += self.grid[ny][nx]
                
                state = self.grid[y][x]
                new_state = state
                
                if state == 1:
                    if count in [2, 3]:
                        new_state = 1
                    else:
                        new_state = 0
                else:
                    if count == 3:
                        new_state = 1
                
                new_grid[y][x] = new_state
                if new_state == 1:
                    population += 1
                if new_state != state:
                    activity += 1
        
        if self.grid == new_grid:
            self.stable_frames += 1
        else:
            self.stable_frames = 0
            
        self.grid = new_grid
        self.generation += 1
        self.population_history.append(population)
        self.activity_history.append(activity)
        
        # Check for reset conditions
        if population == 0 or self.stable_frames > 10 or self.generation >= self.max_generations:
            self.evaluate_and_reset(population)

    def evaluate_and_reset(self, final_pop):
        # Calculate score
        growth_ratio = final_pop / self.initial_pop if self.initial_pop > 0 else 0
        score = growth_ratio * self.generation
        
        # Save if interesting
        if score > 20 or (growth_ratio > 1.2 and self.generation > 50):
            self.save_result(score, final_pop)
            
        self.reset()

    def save_result(self, score, final_pop):
        result = {
            "seed": self.initial_grid,
            "initial_pop": self.initial_pop,
            "final_pop": final_pop,
            "duration": self.generation,
            "score": score,
            "density": self.density,
            "timestamp": time.time()
        }
        
        filename = "parallel_results.json"
        data = []
        if os.path.exists(filename):
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
            except:
                pass
        
        data.append(result)
        # Keep top 100
        data.sort(key=lambda x: x["score"], reverse=True)
        data = data[:100]
        
        with open(filename, "w") as f:
            json.dump(data, f)

    def draw(self, screen):
        # Draw border
        pygame.draw.rect(screen, (50, 50, 50), (self.x_offset, self.y_offset, GAME_WIDTH, GAME_HEIGHT), 1)
        
        # Draw cells
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.grid[y][x] == 1:
                    # Color based on activity/age could be cool, but sticking to white for now
                    rect = pygame.Rect(self.x_offset + x * CELL_SIZE, self.y_offset + y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(screen, WHITE, rect)
        
        # Draw stats overlay
        font = pygame.font.SysFont('Arial', 12)
        stats = f"Gen: {self.generation} Pop: {self.population_history[-1] if self.population_history else 0}"
        text = font.render(stats, True, GREEN)
        screen.blit(text, (self.x_offset + 5, self.y_offset + 5))

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Parallel Game of Life - Evolutionary Search")
    clock = pygame.time.Clock()
    
    games = [MiniGame(i) for i in range(NUM_GAMES)]
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.fill(BLACK)
        
        for game in games:
            game.update()
            game.draw(screen)
            
        pygame.display.flip()
        clock.tick(FPS)
        
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
