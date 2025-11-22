import numpy as np
import random
import json
import os
import time

# Constants
GRID_WIDTH = 32 # Smaller grid for faster training
GRID_HEIGHT = 32
MAX_GENERATIONS = 100
POPULATION_SIZE = 50
LEARNING_RATE = 0.1  # ES learning rate
SIGMA = 0.1  # Noise standard deviation for ES

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
        self.population = []
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
        """Evaluate a network's generated seed through GoL simulation"""
        grid = self.generate_grid(network)
        initial_pop = np.sum(grid)
        
        # Run simulation (NumPy optimized)
        current_grid = grid.copy()
        history_pop = []
        activity_score = 0
        
        # Track initial Center of Mass
        y_indices, x_indices = np.indices(grid.shape)
        
        def get_center_of_mass(g):
            mass = np.sum(g)
            if mass == 0: return None
            cy = np.sum(y_indices * g) / mass
            cx = np.sum(x_indices * g) / mass
            return np.array([cy, cx])

        initial_com = get_center_of_mass(grid)
        final_com = initial_com
        
        for _ in range(MAX_GENERATIONS):
            # Neighbor count
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
            
            pop = np.sum(new_grid)
            history_pop.append(pop)
            
            # Activity: cells that changed state
            activity_score += np.sum(current_grid != new_grid)
            
            if pop == 0 or np.array_equal(current_grid, new_grid):
                break
                
            current_grid = new_grid

        final_pop = history_pop[-1] if history_pop else 0
        duration = len(history_pop)
        
        # Calculate Displacement
        displacement = 0
        if initial_com is not None:
            final_com = get_center_of_mass(current_grid)
            if final_com is not None:
                displacement = np.linalg.norm(final_com - initial_com)
        
        # Calculate Spread (Bounding Box Area)
        spread = 0
        if final_pop > 0:
            rows, cols = np.where(current_grid == 1)
            if len(rows) > 0:
                h = np.max(rows) - np.min(rows) + 1
                w = np.max(cols) - np.min(cols) + 1
                spread = (h * w) / (GRID_WIDTH * GRID_HEIGHT) # Normalized area
        
        # Fitness Function (COMPLEXITY METRIC):
        # 1. Survival (Duration)
        score = duration * 1.0
        
        # 2. Activity (Dynamic behavior)
        score += activity_score * 0.5
        
        # 3. Growth (Expansion)
        if initial_pop > 0:
            growth_ratio = final_pop / initial_pop
            score += growth_ratio * 10.0
            
        # 4. Displacement (Movement/Gliders)
        score += displacement * 5.0
        
        # 5. Spread (Colonization)
        score += spread * 50.0
            
        # 6. Penalty for static/empty
        if initial_pop == 0:
            score -= 100 # Punish death
        elif activity_score < 5:
            score -= 50 # Punish boredom
            
        return max(score, 0)

    def evolve(self):
        """Run one generation of ES optimization"""
        # Sample population from Gaussian around mean
        noise_vectors = []
        networks = []
        
        for _ in range(POPULATION_SIZE):
            noise = np.random.randn(self.genome_size) * SIGMA
            noise_vectors.append(noise)
            
            # Create network with perturbed genome
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
        # Normalize fitness scores
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

    def evolve_step(self, fitness_scores):
        """External evaluation interface for god_massive.py compatibility"""
        # Convert list to numpy array
        fitness_scores = np.array(fitness_scores)
        
        # This assumes population was already generated
        # We need to reconstruct noise_vectors from the population
        # For now, just do standard evolution
        return self.evolve()

    def save(self, filename="god_brain_es.json"):
        """Save the current ES state to disk"""
        data = {
            'mean_genome': self.mean_genome.tolist(),
            'best_genome': self.best_genome.tolist() if self.best_genome is not None else None,
            'best_fitness': float(self.best_fitness),
            'generation': self.generation,
            'fitness_history': self.fitness_history
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved to {filename}")
    
    def load(self, filename="god_brain_es.json"):
        """Load ES state from disk"""
        if not os.path.exists(filename):
            print(f"No save file found at {filename}")
            return False
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.mean_genome = np.array(data['mean_genome'])
        self.mean_network.set_genome(self.mean_genome)
        
        if data['best_genome'] is not None:
            self.best_genome = np.array(data['best_genome'])
        
        self.best_fitness = data['best_fitness']
        self.generation = data['generation']
        self.fitness_history = data['fitness_history']
        
        print(f"Loaded from {filename} - Generation {self.generation}, Best Fitness: {self.best_fitness:.2f}")
        return True


# Alias for backward compatibility
GeneticAlgorithm = EvolutionStrategy

if __name__ == "__main__":
    es = EvolutionStrategy()
    
    # Try to load existing progress
    es.load()
    
    print("Starting Evolution Strategy training...")
    print(f"Grid: {GRID_WIDTH}x{GRID_HEIGHT}, Population: {POPULATION_SIZE}, Sigma: {SIGMA}, LR: {LEARNING_RATE}")
    
    try:
        for gen in range(200):
            es.evolve()
            
            if gen % 10 == 0:
                # Save progress
                es.save()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving progress...")
        es.save()
        print("Done. You can resume by running this script again.")
