import pygame
import numpy as np
import sys
import random
import json
import os

# ==================== CONSTANTS ====================
GRID_WIDTH = 128
GRID_HEIGHT = 128
MAX_GENERATIONS = 100
POPULATION_SIZE = 50
LEARNING_RATE = 0.1  # ES learning rate
SIGMA = 0.1  # Noise standard deviation for ES

# Constants
GAME_SIZE = 128
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60
MAX_AGE = 1200 # Max frames before forced death (20 seconds)
COMPLEXITY_CHECK_INTERVAL = 120 # Check GCM every 120 frames
MIN_COMPLEXITY_THRESHOLD = 40 # Min GCM score to survive (increased from 10)
STAGNATION_THRESHOLD = 50 # Cells must change per frame (increased from 20)
STAGNATION_LIMIT = 10 # Only 10 frames of low activity allowed (reduced from 20)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (50, 255, 50)


# ==================== NEURAL NETWORK ====================
class NeuralNetwork:
    def __init__(self, input_size=4, hidden_size=16, output_size=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Weights and Biases
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.random.randn(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.random.randn(output_size)

    def forward(self, inputs):
        # inputs shape: (N, input_size)
        self.z1 = np.dot(inputs, self.w1) + self.b1
        self.a1 = np.tanh(self.z1) # Tanh activation
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = 1 / (1 + np.exp(-self.z2)) # Sigmoid activation
        return self.a2

    def get_genome(self):
        return np.concatenate([
            self.w1.flatten(),
            self.b1.flatten(),
            self.w2.flatten(),
            self.b2.flatten()
        ])

    def set_genome(self, genome):
        idx = 0
        
        w1_end = idx + self.input_size * self.hidden_size
        self.w1 = genome[idx:w1_end].reshape(self.input_size, self.hidden_size)
        idx = w1_end
        
        b1_end = idx + self.hidden_size
        self.b1 = genome[idx:b1_end]
        idx = b1_end
        
        w2_end = idx + self.hidden_size * self.output_size
        self.w2 = genome[idx:w2_end].reshape(self.hidden_size, self.output_size)
        idx = w2_end
        
        b2_end = idx + self.output_size
        self.b2 = genome[idx:b2_end]

    def mutate(self):
        """Legacy method for compatibility - adds Gaussian noise"""
        genome = self.get_genome()
        noise = np.random.randn(*genome.shape) * 0.1
        self.set_genome(genome + noise)

# ==================== EVOLUTION STRATEGY ====================
class EvolutionStrategy:
    """Natural Evolution Strategies (NES) optimizer"""
    def __init__(self):
        # Mean network (what we're optimizing)
        self.mean_network = NeuralNetwork()
        self.mean_genome = self.mean_network.get_genome()
        self.genome_size = len(self.mean_genome)
        
        self.generation = 0
        self.best_fitness = 0
        self.fitness_history = []
        
        # For compatibility with existing code
        self.best_genome = None
        
        # Precompute coordinates for the grid
        x = np.linspace(-1, 1, GRID_WIDTH)
        y = np.linspace(-1, 1, GRID_HEIGHT)
        xv, yv = np.meshgrid(x, y)
        
        # Calculate distance from center
        dist = np.sqrt(xv**2 + yv**2)
        
        # Flatten and stack: (N, 3) -> x, y, dist
        self.base_inputs = np.column_stack([xv.flatten(), yv.flatten(), dist.flatten()])

    def generate_grid(self, network):
        """Generate a GoL seed grid from a neural network"""
        noise = np.random.randn(self.base_inputs.shape[0], 1) * 0.5
        inputs = np.hstack([self.base_inputs, noise])
        
        # Vectorized forward pass for all cells
        outputs = network.forward(inputs)
        grid = (outputs > 0.5).astype(np.int8).reshape(GRID_HEIGHT, GRID_WIDTH)
        return grid

    def evaluate(self, network):
        """Evaluate a network's generated seed using GoL Complexity Metric (GCM)"""
        grid = self.generate_grid(network)
        
        # Simulate trajectory and collect frames
        frames = [grid.copy()]
        for _ in range(MAX_GENERATIONS):
            current = frames[-1]
            
            # Game of Life step
            N = (np.roll(current, 1, axis=0) + np.roll(current, -1, axis=0) +
                 np.roll(current, 1, axis=1) + np.roll(current, -1, axis=1) +
                 np.roll(np.roll(current, 1, axis=0), 1, axis=1) +
                 np.roll(np.roll(current, 1, axis=0), -1, axis=1) +
                 np.roll(np.roll(current, -1, axis=0), 1, axis=1) +
                 np.roll(np.roll(current, -1, axis=0), -1, axis=1))
            
            birth = (N == 3) & (current == 0)
            survive = ((N == 2) | (N == 3)) & (current == 1)
            new_grid = np.zeros_like(current)
            new_grid[birth | survive] = 1
            
            frames.append(new_grid)
            
            # Early stopping
            if np.sum(new_grid) == 0:
                break
            if len(frames) > 2 and np.array_equal(frames[-1], frames[-2]):
                break
        
        # Compute GCM
        return self.gol_complexity_metric(frames)
    
    # ==================== GCM HELPER FUNCTIONS ====================
    
    def cell_density(self, F):
        """Fraction of alive cells"""
        return np.sum(F) / F.size
    
    def activity(self, F_prev, F_curr):
        """Fraction of cells that changed"""
        return np.sum(F_prev != F_curr) / F_prev.size
    
    def extract_patches(self, F, k=8):
        """Extract all k x k non-overlapping patches"""
        H, W = F.shape
        patches = []
        for y in range(0, H, k):
            for x in range(0, W, k):
                if y + k <= H and x + k <= W:
                    patch = F[y:y+k, x:x+k]
                    patches.append(tuple(patch.flatten().tolist()))
        return patches
    
    def patch_uniqueness(self, F, k=8):
        """Unique patch fraction (diversity)"""
        patches = self.extract_patches(F, k)
        if not patches:
            return 0.0
        return len(set(patches)) / len(patches)
    
    def patch_type_set(self, F, k=8):
        """Set of unique patch types"""
        return set(self.extract_patches(F, k))
    
    def patch_overlap(self, F_prev, F_curr, k=8):
        """Jaccard overlap of patch types between frames"""
        A = self.patch_type_set(F_prev, k)
        B = self.patch_type_set(F_curr, k)
        if not A and not B:
            return 1.0
        if not A or not B:
            return 0.0
        inter = len(A & B)
        union = len(A | B)
        return inter / union
    
    # ==================== GOODNESS FUNCTIONS ====================
    
    def density_goodness(self, rho):
        """Reward mid-density (parabola peaked at 0.5)"""
        return 4 * rho * (1 - rho)
    
    def activity_goodness(self, a, mu=0.2, sigma=0.1):
        """Reward moderate activity (Gaussian)"""
        if sigma <= 0:
            return 0.0
        z = (a - mu) / sigma
        return np.exp(-0.5 * z * z)
    
    def type_diversity_goodness(self, u):
        """Reward mid-range uniqueness"""
        return 4 * u * (1 - u)
    
    def overlap_goodness(self, o):
        """Reward high motif persistence"""
        return o
    
    def frame_complexity(self, F_prev, F_curr, k=8,
                        alpha=0.2, beta=0.3, gamma=0.3, delta=0.2):
        """Compute complexity score for a single frame transition"""
        rho = self.cell_density(F_curr)
        a = self.activity(F_prev, F_curr)
        u = self.patch_uniqueness(F_curr, k)
        o = self.patch_overlap(F_prev, F_curr, k)
        
        g_cells = self.density_goodness(rho)
        g_activity = self.activity_goodness(a)
        g_types = self.type_diversity_goodness(u)
        g_overlap = self.overlap_goodness(o)
        
        return (alpha * g_cells +
                beta * g_activity +
                gamma * g_types +
                delta * g_overlap)
    
    def gol_complexity_metric(self, frames, k=8,
                             alpha=0.2, beta=0.3, gamma=0.3, delta=0.2,
                             burnin_fraction=1/3):
        """
        Compute GoL Complexity Metric (GCM) for entire trajectory.
        
        Returns a scalar in [0,1] that is:
        - Low for trivial worlds (empty, still-lifes)
        - Low-medium for chaos/noise
        - High for structured dynamics (gliders, guns, circuits)
        """
        T = len(frames) - 1
        if T <= 0:
            return 0.0
        
        B = int(T * burnin_fraction)  # Skip early transients
        scores = []
        
        for t in range(max(1, B), T + 1):
            F_prev = frames[t-1]
            F_curr = frames[t]
            s = self.frame_complexity(F_prev, F_curr, k,
                                     alpha=alpha, beta=beta,
                                     gamma=gamma, delta=delta)
            scores.append(s)
        
        if not scores:
            return 0.0
        
        # Average complexity over non-burnin frames
        avg_complexity = np.mean(scores)
        
        # Scale to make scores more meaningful (0-100 range)
        return avg_complexity * 100

    def evolve(self):
        """Run one generation of ES optimization"""
        # Sample population from Gaussian around mean
        noise_vectors = []
        networks = []
        
        for _ in range(POPULATION_SIZE):
            noise = np.random.randn(self.genome_size) * SIGMA
            noise_vectors.append(noise)
            
            network = NeuralNetwork()
            network.set_genome(self.mean_genome + noise)
            networks.append(network)
        
        # Evaluate all networks
        fitness_scores = np.array([self.evaluate(net) for net in networks])
        
        # Track best
        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = fitness_scores[best_idx]
            self.best_genome = self.mean_genome + noise_vectors[best_idx]
        
        # Fitness-weighted gradient estimation
        if np.std(fitness_scores) > 0:
            normalized_fitness = (fitness_scores - np.mean(fitness_scores)) / (np.std(fitness_scores) + 1e-8)
        else:
            normalized_fitness = fitness_scores
        
        # Estimate gradient
        gradient = np.zeros(self.genome_size)
        for i in range(POPULATION_SIZE):
            gradient += normalized_fitness[i] * noise_vectors[i]
        gradient /= (POPULATION_SIZE * SIGMA)
        
        # Update mean genome
        self.mean_genome += LEARNING_RATE * gradient
        self.mean_network.set_genome(self.mean_genome)
        
        # Update history
        avg_fitness = np.mean(fitness_scores)
        self.fitness_history.append({
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'avg_fitness': avg_fitness
        })
        
        self.generation += 1
        
        print(f"Gen {self.generation}: Best Fitness = {self.best_fitness:.2f}, Avg = {avg_fitness:.2f}")
        
        return self.best_fitness

    def save(self, filename="god_brain_es.json", hall_of_fame=None):
        """Save the current ES state to disk"""
        data = {
            'mean_genome': self.mean_genome.tolist(),
            'best_genome': self.best_genome.tolist() if self.best_genome is not None else None,
            'best_fitness': float(self.best_fitness),
            'generation': self.generation,
            'fitness_history': self.fitness_history,
            'hall_of_fame': [(float(score), genome.tolist()) for score, genome in hall_of_fame] if hall_of_fame else []
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved to {filename} (HoF: {len(hall_of_fame) if hall_of_fame else 0} genomes)")
    
    def load(self, filename="god_brain_es.json"):
        """Load ES state from disk"""
        if not os.path.exists(filename):
            print(f"No save file found at {filename}")
            return False, []
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.mean_genome = np.array(data['mean_genome'])
        self.mean_network.set_genome(self.mean_genome)
        
        if data['best_genome'] is not None:
            self.best_genome = np.array(data['best_genome'])
        
        self.best_fitness = data['best_fitness']
        self.generation = data['generation']
        self.fitness_history = data['fitness_history']
        
        # Load Hall of Fame
        hall_of_fame = []
        if 'hall_of_fame' in data:
            hall_of_fame = [(score, np.array(genome)) for score, genome in data['hall_of_fame']]
        
        print(f"Loaded from {filename} - Gen {self.generation}, Best: {self.best_fitness:.2f}, HoF: {len(hall_of_fame)}")
        return True, hall_of_fame

# ==================== MASSIVE GOD MODE ====================
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
        self.stats_age = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.stats_stagnation = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.stats_initial_pop = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.stats_activity_accum = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.stats_alive = np.ones((self.rows, self.cols), dtype=bool)
        self.stats_last_gcm = np.zeros((self.rows, self.cols), dtype=np.float32)
        
        # Frame history for GCM calculation (store last N frames per game)
        self.frame_history = [[[] for _ in range(self.cols)] for _ in range(self.rows)]
        
        # Population Management
        self.hall_of_fame = []
        self.best_fitness_ever = 0
        
        # Active Genomes Grid
        self.active_genomes = [[NeuralNetwork() for _ in range(self.cols)] for _ in range(self.rows)]
        
        # Isolation Mask (Void Barriers)
        self.isolation_mask = np.ones((self.sim_height, self.sim_width), dtype=np.int8)
        
        # Create dead zones at borders
        for c in range(self.cols + 1):
            x = c * GAME_SIZE
            if x < self.sim_width:
                self.isolation_mask[:, x] = 0
            if x - 1 >= 0:
                self.isolation_mask[:, x-1] = 0
                
        for r in range(self.rows + 1):
            y = r * GAME_SIZE
            if y < self.sim_height:
                self.isolation_mask[y, :] = 0
            if y - 1 >= 0:
                self.isolation_mask[y-1, :] = 0
                
        # Pre-render Grid Overlay
        self.grid_overlay = pygame.Surface((self.sim_width, self.sim_height), pygame.SRCALPHA)
        grid_color = (30, 30, 30)
        
        for c in range(1, self.cols):
            x = c * GAME_SIZE
            pygame.draw.line(self.grid_overlay, grid_color, (x, 0), (x, self.sim_height))
            
        for r in range(1, self.rows):
            y = r * GAME_SIZE
            pygame.draw.line(self.grid_overlay, grid_color, (0, y), (self.sim_width, y))
        
        # Initialize all games
        print("Initializing population...")
        for r in range(self.rows):
            for c in range(self.cols):
                self.reset_game(r, c, initial=True)

    def reset_game(self, r, c, initial=False, target_grid=None):
        child = NeuralNetwork()
        
        if not initial and len(self.hall_of_fame) > 2:
            parents = random.sample(self.hall_of_fame, min(len(self.hall_of_fame), 5))
            parents.sort(key=lambda x: x[0], reverse=True)
            parent_genome = parents[0][1]
            
            child.set_genome(parent_genome)
            child.mutate()
        elif not initial:
             if self.es.best_genome is not None:
                 child.set_genome(self.es.best_genome)
                 child.mutate()
        
        self.active_genomes[r][c] = child
        seed = self.es.generate_grid(child)
        
        y = r * GAME_SIZE
        x = c * GAME_SIZE
        
        if target_grid is None:
            target_grid = self.grid

        target_grid[y:y+GAME_SIZE, x:x+GAME_SIZE] = seed
        
        self.stats_age[r, c] = 0
        self.stats_stagnation[r, c] = 0
        self.stats_initial_pop[r, c] = np.sum(seed)
        self.stats_activity_accum[r, c] = 0
        self.stats_alive[r, c] = True
        self.stats_last_gcm[r, c] = 0.0
        self.frame_history[r][c] = [seed.copy()]

    def update(self):
        # Game of Life Update
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
        
        new_grid &= self.isolation_mask
        
        # Calculate Metrics
        changed = (current_grid != new_grid).astype(np.int32)
        reshaped_changed = changed.reshape(self.rows, GAME_SIZE, self.cols, GAME_SIZE)
        activity_per_game = reshaped_changed.sum(axis=(1, 3))
        
        reshaped_new = new_grid.reshape(self.rows, GAME_SIZE, self.cols, GAME_SIZE)
        pop_per_game = reshaped_new.sum(axis=(1, 3))
        
        # Update Stats
        self.stats_age += 1
        self.stats_activity_accum += activity_per_game
        
        # Update frame history for ALL games
        for r in range(self.rows):
            for c in range(self.cols):
                y, x = r * GAME_SIZE, c * GAME_SIZE
                game_grid = new_grid[y:y+GAME_SIZE, x:x+GAME_SIZE].copy()
                
                # Keep last 30 frames for GCM calculation
                self.frame_history[r][c].append(game_grid)
                if len(self.frame_history[r][c]) > 30:
                    self.frame_history[r][c].pop(0)
        
        # Calculate GCM for one game per frame (rolling)
        if hasattr(self, 'frame_count'):
            total_games = self.rows * self.cols
            game_index = self.frame_count % total_games
            r = game_index // self.cols
            c = game_index % self.cols
            
            # Calculate GCM if game has enough frames
            if len(self.frame_history[r][c]) >= 10:
                gcm = self.es.gol_complexity_metric(self.frame_history[r][c], k=8)
                self.stats_last_gcm[r, c] = gcm
        
        # Identify games to reset based on GCM scores and basic conditions
        dead_mask = (pop_per_game == 0)  # Truly dead (no cells)
        old_mask = (self.stats_age > MAX_AGE)  # Too old
        
        # GCM-based reset: Games with low complexity OR that haven't been evaluated yet but are old enough
        low_complexity_mask = np.zeros((self.rows, self.cols), dtype=bool)
        for r in range(self.rows):
            for c in range(self.cols):
                # If game is old enough and has a GCM score
                if self.stats_age[r, c] >= 100:  # At least 100 frames old
                    gcm_score = self.stats_last_gcm[r, c]
                    if gcm_score > 0 and gcm_score < MIN_COMPLEXITY_THRESHOLD:
                        low_complexity_mask[r, c] = True
        
        reset_mask = dead_mask | old_mask | low_complexity_mask
        
        # Process Resets
        rows_to_reset, cols_to_reset = np.where(reset_mask)
        
        if hasattr(self, 'frame_count') and self.frame_count % 60 == 0 and len(rows_to_reset) > 0:
            # Count reasons for reset
            dead_count = np.sum(dead_mask)
            old_count = np.sum(old_mask)
            low_complex_count = np.sum(low_complexity_mask)
            print(f"Reset Queue: {len(rows_to_reset)} (Dead:{dead_count} Old:{old_count} LowGCM:{low_complex_count})")
        elif not hasattr(self, 'frame_count'):
             self.frame_count = 0
        self.frame_count += 1

        # Process ALL resets immediately (no limit)
        if len(rows_to_reset) > 0:
            for i in range(len(rows_to_reset)):
                r, c = rows_to_reset[i], cols_to_reset[i]
                
                # Calculate final GCM score for Hall of Fame
                frames = self.frame_history[r][c]
                if len(frames) >= 5:
                    gcm_score = self.es.gol_complexity_metric(frames, k=8)
                else:
                    gcm_score = 0.0
                
                # Add to Hall of Fame if complex enough
                if gcm_score > MIN_COMPLEXITY_THRESHOLD:
                    genome = self.active_genomes[r][c].get_genome()
                    self.hall_of_fame.append((gcm_score, genome))
                    if len(self.hall_of_fame) > 100:
                        self.hall_of_fame.sort(key=lambda x: x[0], reverse=True)
                        self.hall_of_fame = self.hall_of_fame[:50]
                
                if gcm_score > self.best_fitness_ever:
                    self.best_fitness_ever = gcm_score
                
                self.reset_game(r, c, target_grid=new_grid)
        
        self.grid = new_grid

    def draw(self):
        # Create RGB array
        rgb_array = np.zeros((self.sim_height, self.sim_width, 3), dtype=np.uint8)
        
        # Normalize GCM scores relative to current frame (min-max normalization)
        active_scores = self.stats_last_gcm[self.stats_last_gcm > 0]
        if len(active_scores) > 0:
            min_gcm = np.min(active_scores)
            max_gcm = np.max(active_scores)
            
            # Avoid division by zero
            if max_gcm - min_gcm < 0.1:
                normalized_scores = np.ones_like(self.stats_last_gcm) * 0.5
            else:
                # Normalize to 0-1 range based on current min/max
                normalized_scores = np.clip((self.stats_last_gcm - min_gcm) / (max_gcm - min_gcm), 0, 1)
        else:
            normalized_scores = np.zeros_like(self.stats_last_gcm)
        
        # Smooth continuous color gradient: Blue -> Cyan -> Green -> Yellow -> Orange -> Red
        for r in range(self.rows):
            for c in range(self.cols):
                y_start = r * GAME_SIZE
                x_start = c * GAME_SIZE
                
                # Get normalized score (relative to current frame)
                t = normalized_scores[r, c]
                
                # Convert from a smooth blue-to-red spectrum
                if t < 0.2:  # Deep blue to cyan
                    ratio = t / 0.2
                    color = np.array([0, int(ratio * 255), 255])
                elif t < 0.4:  # Cyan to green
                    ratio = (t - 0.2) / 0.2
                    color = np.array([0, 255, int((1 - ratio) * 255)])
                elif t < 0.6:  # Green to yellow
                    ratio = (t - 0.4) / 0.2
                    color = np.array([int(ratio * 255), 255, 0])
                elif t < 0.8:  # Yellow to orange
                    ratio = (t - 0.6) / 0.2
                    color = np.array([255, int((1 - ratio * 0.5) * 255), 0])
                else:  # Orange to red
                    ratio = (t - 0.8) / 0.2
                    color = np.array([255, int((1 - ratio) * 128), 0])
                
                # Apply to this game's region (vectorized)
                game_mask = self.grid[y_start:y_start+GAME_SIZE, x_start:x_start+GAME_SIZE]
                rgb_array[y_start:y_start+GAME_SIZE, x_start:x_start+GAME_SIZE][game_mask == 1] = color
        
        # Transpose for pygame (pygame expects width, height order)
        rgb_array = np.transpose(rgb_array, (1, 0, 2))
        
        pygame.surfarray.blit_array(self.screen, rgb_array)
        self.screen.blit(self.grid_overlay, (0, 0))
        
        # Update text to show GCM range
        active_scores = self.stats_last_gcm[self.stats_last_gcm > 0]
        if len(active_scores) > 0:
            min_gcm_val = np.min(active_scores)
            max_gcm_val = np.max(active_scores)
            avg_gcm = np.mean(active_scores)
        else:
            min_gcm_val = max_gcm_val = avg_gcm = 0
        
        text = self.font.render(f"Best: {self.best_fitness_ever:.1f} | Range: {min_gcm_val:.1f}-{max_gcm_val:.1f} | Avg: {avg_gcm:.1f} | HoF: {len(self.hall_of_fame)}", True, WHITE)
        self.screen.blit(text, (20, 20))
        
        pygame.display.flip()

    def run(self):
        # Load ES state and Hall of Fame
        success, loaded_hof = self.es.load()
        if loaded_hof:
            self.hall_of_fame = loaded_hof
            self.best_fitness_ever = max([score for score, _ in loaded_hof]) if loaded_hof else 0
            print(f"Restored Hall of Fame with {len(loaded_hof)} patterns, best: {self.best_fitness_ever:.2f}")
        
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
                if frame_counter % 600 == 0:
                    self.es.save(hall_of_fame=self.hall_of_fame)
        except KeyboardInterrupt:
            print("\nSaving progress...")
            self.es.save(hall_of_fame=self.hall_of_fame)

        self.es.save(hall_of_fame=self.hall_of_fame)
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    MassiveGodMode().run()
