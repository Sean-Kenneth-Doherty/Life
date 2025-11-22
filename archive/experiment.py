import random
import json
import time

# Constants matching main.py
GRID_WIDTH = 80
GRID_HEIGHT = 60
MAX_GENERATIONS = 100
NUM_EXPERIMENTS = 10

def run_simulation(seed_grid):
    grid = [row[:] for row in seed_grid]
    history = []
    
    initial_pop = sum(sum(row) for row in grid)
    if initial_pop == 0:
        return 0, 0, 0 # Dead on arrival

    for gen in range(MAX_GENERATIONS):
        new_grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        population = 0
        activity = 0
        
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                # Count neighbors
                count = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                            count += grid[ny][nx]
                
                state = grid[y][x]
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
        
        if grid == new_grid: # Stabilized
            return population, gen, activity
            
        grid = new_grid
        
        if population == 0: # Extinct
            return 0, gen, 0

    return population, MAX_GENERATIONS, activity

def generate_random_grid(density=0.5):
    return [[1 if random.random() < density else 0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

def main():
    print(f"Running {NUM_EXPERIMENTS} experiments...")
    results = []
    
    start_time = time.time()
    
    for i in range(NUM_EXPERIMENTS):
        density = random.uniform(0.1, 0.6) # Vary density
        seed = generate_random_grid(density)
        initial_pop = sum(sum(row) for row in seed)
        
        final_pop, duration, final_activity = run_simulation(seed)
        
        # Growth Score: Reward population growth and long duration
        # Avoid division by zero
        growth_ratio = final_pop / initial_pop if initial_pop > 0 else 0
        score = growth_ratio * duration
        
        # Filter for "interesting" results
        # e.g., Population grew significantly OR it lasted a long time with activity
        if score > 10 or (growth_ratio > 1.0 and duration > 20):
            results.append({
                "seed": seed,
                "initial_pop": initial_pop,
                "final_pop": final_pop,
                "duration": duration,
                "score": score,
                "density": density
            })
            
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{NUM_EXPERIMENTS}...")

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Save top 10
    top_results = results[:20]
    
    with open("results.json", "w") as f:
        json.dump(top_results, f)
        
    print(f"Done! Found {len(results)} interesting seeds. Saved top {len(top_results)} to results.json.")
    print(f"Time taken: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
