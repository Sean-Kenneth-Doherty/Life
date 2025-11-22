import pygame
import sys
import random
import json
import os

# Constants
CELL_SIZE = 10
GRID_WIDTH = 80
GRID_HEIGHT = 60
SIDEBAR_WIDTH = 300
SCREEN_WIDTH = (GRID_WIDTH * CELL_SIZE) + SIDEBAR_WIDTH
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE
FPS = 10

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
BLUE = (50, 50, 255)
DARK_GRAY = (40, 40, 40)

class GameOfLife:
    def __init__(self):
        self.grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.running = False
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Conway's Game of Life - Metrics & Randomization")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 16)
        
        # Metrics
        self.generation = 0
        self.population_history = []
        self.activity_history = []
        self.max_history = 200
        
        # Replay
        self.replay_data = []
        self.current_replay_index = 0
        self.load_replay_data()

    def load_replay_data(self):
        if os.path.exists("results.json"):
            with open("results.json", "r") as f:
                self.replay_data = json.load(f)

    def draw_grid(self):
        # Draw Game Area
        pygame.draw.rect(self.screen, BLACK, (0, 0, GRID_WIDTH * CELL_SIZE, SCREEN_HEIGHT))
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.grid[y][x] == 1:
                    rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(self.screen, WHITE, rect)

    def draw_metrics(self):
        # Draw Sidebar
        sidebar_rect = pygame.Rect(GRID_WIDTH * CELL_SIZE, 0, SIDEBAR_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, DARK_GRAY, sidebar_rect)
        
        # Text Stats
        stats = [
            f"Generation: {self.generation}",
            f"Population: {self.population_history[-1] if self.population_history else 0}",
            f"Activity: {self.activity_history[-1] if self.activity_history else 0}",
            "",
            "Controls:",
            "Space: Pause/Resume",
            "C: Clear",
            "R: Random (50%)",
            "1-9: Random (10-90%)",
            "L: Load Evolved Seed",
            "Click: Toggle Cell"
        ]
        
        if self.replay_data:
             stats.insert(4, f"Replay Available: {len(self.replay_data)}")
        
        y_offset = 20
        for line in stats:
            text = self.font.render(line, True, WHITE)
            self.screen.blit(text, (GRID_WIDTH * CELL_SIZE + 10, y_offset))
            y_offset += 25

        # Draw Graphs
        self.draw_graph("Population", self.population_history, GREEN, y_offset + 20)
        self.draw_graph("Activity", self.activity_history, RED, y_offset + 150)

    def draw_graph(self, title, data, color, y_pos):
        graph_height = 100
        graph_width = SIDEBAR_WIDTH - 20
        x_start = GRID_WIDTH * CELL_SIZE + 10
        
        # Title
        title_surf = self.font.render(title, True, color)
        self.screen.blit(title_surf, (x_start, y_pos - 20))
        
        # Background
        bg_rect = pygame.Rect(x_start, y_pos, graph_width, graph_height)
        pygame.draw.rect(self.screen, BLACK, bg_rect)
        pygame.draw.rect(self.screen, GRAY, bg_rect, 1)
        
        if not data:
            return

        # Normalize and draw lines
        max_val = max(data) if max(data) > 0 else 1
        points = []
        for i, val in enumerate(data[-graph_width:]): # Show last N points that fit
            x = x_start + i
            y = y_pos + graph_height - (val / max_val * graph_height)
            points.append((x, y))
        
        if len(points) > 1:
            pygame.draw.lines(self.screen, color, False, points, 2)

    def update_grid(self):
        if not self.running:
            return

        new_grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        activity = 0
        population = 0
        
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                neighbors = self.count_neighbors(x, y)
                state = self.grid[y][x]
                new_state = state
                
                if state == 1:
                    if neighbors in [2, 3]:
                        new_state = 1
                    else:
                        new_state = 0
                else:
                    if neighbors == 3:
                        new_state = 1
                
                new_grid[y][x] = new_state
                
                if new_state == 1:
                    population += 1
                if new_state != state:
                    activity += 1
                    
        self.grid = new_grid
        self.generation += 1
        self.population_history.append(population)
        self.activity_history.append(activity)
        
        # Trim history
        if len(self.population_history) > self.max_history:
            self.population_history.pop(0)
            self.activity_history.pop(0)

    def count_neighbors(self, x, y):
        count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                    count += self.grid[ny][nx]
        return count

    def randomize_grid(self, density=0.5):
        self.grid = [[1 if random.random() < density else 0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.generation = 0
        self.population_history = []
        self.activity_history = []

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                if x < GRID_WIDTH * CELL_SIZE: # Only click in grid area
                    grid_x, grid_y = x // CELL_SIZE, y // CELL_SIZE
                    self.grid[grid_y][grid_x] = 1 - self.grid[grid_y][grid_x]
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.running = not self.running
                elif event.key == pygame.K_c:
                    self.grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
                    self.running = False
                    self.generation = 0
                    self.population_history = []
                    self.activity_history = []
                elif event.key == pygame.K_r:
                    self.randomize_grid(0.5)
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    density = (event.key - pygame.K_0) / 10.0
                    self.randomize_grid(density)
                elif event.key == pygame.K_l:
                    self.load_next_replay()

    def load_next_replay(self):
        if not self.replay_data:
            print("No replay data found!")
            return
            
        data = self.replay_data[self.current_replay_index]
        self.grid = data["seed"]
        self.generation = 0
        self.population_history = []
        self.activity_history = []
        print(f"Loaded seed with Score: {data['score']:.2f}, Duration: {data['duration']}")
        
        self.current_replay_index = (self.current_replay_index + 1) % len(self.replay_data)

    def run(self):
        while True:
            self.handle_input()
            self.update_grid()
            self.screen.fill(BLACK) # Clear screen
            self.draw_grid()
            self.draw_metrics()
            pygame.display.flip()
            self.clock.tick(FPS)

if __name__ == "__main__":
    pygame.init()
    game = GameOfLife()
    game.run()
