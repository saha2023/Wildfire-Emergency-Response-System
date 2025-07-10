import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import time
import random
import itertools
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Advanced Algorithm Comparison System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .algorithm-comparison {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .priority-section {
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #fafafa;
    }
    
    .priority-critical {
        border-color: #f44336;
        background-color: #ffebee;
    }
    
    .priority-medium {
        border-color: #ff9800;
        background-color: #fff3e0;
    }
    
    .priority-low {
        border-color: #4caf50;
        background-color: #e8f5e8;
    }
    
    .priority-high {
        background-color: #ffebee;
        color: #c62828;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .algorithm-card {
        background: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .genetic-card {
        border-color: #9C27B0;
        background-color: #f3e5f5;
    }
    
    .heuristic-card {
        border-color: #FF9800;
        background-color: #fff3e0;
    }
    
    .brute-force-card {
        border-color: #4CAF50;
        background-color: #e8f5e8;
    }
    
    .dp-card {
        border-color: #2196F3;
        background-color: #e3f2fd;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'data_loaded': False,
        'priorities_set': False,
        'last_manual_refresh': datetime.now(),
        'base_data': None,
        'convex_hull_calculated': False,
        'algorithms_calculated': False,
        'selected_priorities_for_analysis': ['Critical', 'Medium', 'Low'],
        'priority_results': {
            'convex_hulls': {},
            'algorithm_results': {}
        }
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Algorithm implementations
class SafeStringMatcher:
    @staticmethod
    def safe_kmp_search(text, pattern):
        if not text or not pattern:
            return []
        
        try:
            matches = []
            text_lower = text.lower()
            pattern_lower = pattern.lower()
            
            start = 0
            while True:
                pos = text_lower.find(pattern_lower, start)
                if pos == -1:
                    break
                matches.append(pos)
                start = pos + 1
            return matches
        except:
            return []

class SafeConvexHull:
    @staticmethod
    def safe_graham_scan(points):
        if len(points) < 3:
            return points
        
        try:
            def cross_product(o, a, b):
                return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            
            points = list(set(points))
            if len(points) < 3:
                return points
                
            points.sort()
            
            lower = []
            for p in points:
                while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
                    lower.pop()
                lower.append(p)
            
            upper = []
            for p in reversed(points):
                while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
                    upper.pop()
                upper.append(p)
            
            return lower[:-1] + upper[:-1]
        except:
            return points

class TwoOptHeuristic:
    """2-opt heuristic algorithm for TSP optimization"""
    
    def __init__(self, max_iterations=1000):
        self.max_iterations = max_iterations
        self.petrol_efficiency = 8.0  # km per liter
        self.petrol_cost_per_liter = 1.5  # cost per liter
    
    @staticmethod
    def calculate_distance(point1, point2):
        """Calculate Euclidean distance between two points"""
        try:
            return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        except:
            return float('inf')
    
    def calculate_route_distance(self, route, points):
        """Calculate total distance of the route"""
        if len(route) < 2:
            return 0
        
        total_distance = 0
        for i in range(len(route)):
            from_idx = route[i]
            to_idx = route[(i + 1) % len(route)]
            total_distance += self.calculate_distance(points[from_idx], points[to_idx])
        return total_distance
    
    def calculate_petrol_cost(self, route, points):
        """Calculate petrol cost for the route"""
        distance = self.calculate_route_distance(route, points)
        petrol_needed = distance / self.petrol_efficiency
        return petrol_needed * self.petrol_cost_per_liter
    
    def two_opt_swap(self, route, i, k):
        """Perform 2-opt swap on route"""
        new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
        return new_route
    
    def solve_tsp(self, points):
        """Solve TSP using 2-opt heuristic"""
        if len(points) < 2:
            return [], [], 0, [], {}
        
        try:
            n = len(points)
            
            # Start with nearest neighbor heuristic
            current_route = list(range(n))
            current_distance = self.calculate_route_distance(current_route, points)
            initial_distance = current_distance
            initial_cost = self.calculate_petrol_cost(current_route, points)
            
            # Track improvements
            improvement_count = 0
            iterations = 0
            distance_history = [current_distance]
            
            # 2-opt improvement
            improved = True
            while improved and iterations < self.max_iterations:
                improved = False
                iterations += 1
                
                for i in range(1, n - 1):
                    for k in range(i + 1, n):
                        if k - i == 1:
                            continue  # Skip adjacent edges
                        
                        new_route = self.two_opt_swap(current_route, i, k)
                        new_distance = self.calculate_route_distance(new_route, points)
                        
                        if new_distance < current_distance:
                            current_route = new_route
                            current_distance = new_distance
                            improvement_count += 1
                            improved = True
                            distance_history.append(current_distance)
                            break
                    if improved:
                        break
            
            final_cost = self.calculate_petrol_cost(current_route, points)
            improvement_distance = ((initial_distance - current_distance) / initial_distance * 100) if initial_distance > 0 else 0
            improvement_cost = ((initial_cost - final_cost) / initial_cost * 100) if initial_cost > 0 else 0
            
            # Create path coordinates
            path_coordinates = [points[i] for i in current_route]
            
            # Create edges
            edges = [(current_route[i], current_route[(i + 1) % len(current_route)]) for i in range(len(current_route))]
            
            results = {
                'algorithm': '2-opt Heuristic',
                'best_route': current_route,
                'path_coordinates': path_coordinates,
                'edges': edges,
                'total_distance': current_distance,
                'total_cost': final_cost,
                'initial_distance': initial_distance,
                'initial_cost': initial_cost,
                'improvement_distance': improvement_distance,
                'improvement_cost': improvement_cost,
                'iterations': iterations,
                'improvements': improvement_count,
                'distance_history': distance_history,
                'petrol_liters': current_distance / self.petrol_efficiency,
                'execution_time': 0  # Will be set by caller
            }
            
            return edges, path_coordinates, current_distance, current_route, results
            
        except Exception as e:
            return [], [], 0, [], {}

class GeneticAlgorithmTSP:
    """Genetic Algorithm for TSP optimization"""
    
    def __init__(self, population_size=100, elite_size=20, mutation_rate=0.01, generations=300):
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.petrol_efficiency = 8.0
        self.petrol_cost_per_liter = 1.5
    
    @staticmethod
    def calculate_distance(point1, point2):
        """Calculate Euclidean distance between two points"""
        try:
            return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        except:
            return float('inf')
    
    def calculate_route_distance(self, route, points):
        """Calculate total distance of the route"""
        if len(route) < 2:
            return 0
        
        total_distance = 0
        for i in range(len(route)):
            from_idx = route[i]
            to_idx = route[(i + 1) % len(route)]
            total_distance += self.calculate_distance(points[from_idx], points[to_idx])
        return total_distance
    
    def calculate_petrol_cost(self, route, points):
        """Calculate petrol cost for the route"""
        distance = self.calculate_route_distance(route, points)
        petrol_needed = distance / self.petrol_efficiency
        return petrol_needed * self.petrol_cost_per_liter
    
    def fitness(self, route, points):
        """Fitness function - inverse of petrol cost"""
        cost = self.calculate_petrol_cost(route, points)
        return 1 / (cost + 1e-10)
    
    def create_route(self, city_count):
        """Create a random route"""
        return list(range(city_count))
    
    def initial_population(self, city_count):
        """Create initial population"""
        population = []
        for _ in range(self.population_size):
            route = self.create_route(city_count)
            random.shuffle(route)
            population.append(route)
        return population
    
    def rank_routes(self, population, points):
        """Rank routes based on fitness"""
        fitness_results = []
        for i, route in enumerate(population):
            fitness_value = self.fitness(route, points)
            fitness_results.append((i, fitness_value))
        return sorted(fitness_results, key=lambda x: x[1], reverse=True)
    
    def selection(self, ranked_pop):
        """Tournament selection"""
        selection_results = []
        
        for i in range(self.elite_size):
            selection_results.append(ranked_pop[i][0])
        
        for _ in range(len(ranked_pop) - self.elite_size):
            tournament_size = 5
            tournament = random.sample(ranked_pop, min(tournament_size, len(ranked_pop)))
            winner = max(tournament, key=lambda x: x[1])
            selection_results.append(winner[0])
        
        return selection_results
    
    def create_mating_pool(self, population, selection_results):
        """Create mating pool"""
        return [population[i] for i in selection_results]
    
    def order_crossover(self, parent1, parent2):
        """Order crossover for TSP"""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child = [-1] * size
        child[start:end] = parent1[start:end]
        
        pointer = end
        for city in parent2[end:] + parent2[:end]:
            if city not in child:
                child[pointer % size] = city
                pointer += 1
        
        return child
    
    def breed_population(self, mating_pool):
        """Create next generation"""
        children = []
        
        for i in range(self.elite_size):
            children.append(mating_pool[i])
        
        for _ in range(len(mating_pool) - self.elite_size):
            parent1 = random.choice(mating_pool[:self.elite_size])
            parent2 = random.choice(mating_pool[:self.elite_size])
            child = self.order_crossover(parent1, parent2)
            children.append(child)
        
        return children
    
    def mutate(self, individual):
        """Swap mutation"""
        mutated = individual.copy()
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(mutated)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated
    
    def mutate_population(self, population):
        """Apply mutation to population"""
        return [self.mutate(individual) for individual in population]
    
    def evolve_generation(self, population, points):
        """Evolve one generation"""
        ranked_pop = self.rank_routes(population, points)
        selection_results = self.selection(ranked_pop)
        mating_pool = self.create_mating_pool(population, selection_results)
        children = self.breed_population(mating_pool)
        next_generation = self.mutate_population(children)
        return next_generation
    
    def solve_tsp(self, points):
        """Solve TSP using genetic algorithm"""
        if len(points) < 2:
            return [], [], 0, [], {}
        
        try:
            city_count = len(points)
            population = self.initial_population(city_count)
            
            best_costs = []
            
            initial_ranked = self.rank_routes(population, points)
            initial_best = population[initial_ranked[0][0]]
            initial_distance = self.calculate_route_distance(initial_best, points)
            initial_cost = self.calculate_petrol_cost(initial_best, points)
            
            for generation in range(self.generations):
                population = self.evolve_generation(population, points)
                
                ranked_pop = self.rank_routes(population, points)
                best_route = population[ranked_pop[0][0]]
                best_costs.append(self.calculate_petrol_cost(best_route, points))
            
            final_ranked = self.rank_routes(population, points)
            best_route = population[final_ranked[0][0]]
            final_distance = self.calculate_route_distance(best_route, points)
            final_cost = self.calculate_petrol_cost(best_route, points)
            
            improvement_distance = ((initial_distance - final_distance) / initial_distance * 100) if initial_distance > 0 else 0
            improvement_cost = ((initial_cost - final_cost) / initial_cost * 100) if initial_cost > 0 else 0
            
            path_coordinates = [points[i] for i in best_route]
            edges = [(best_route[i], best_route[(i + 1) % len(best_route)]) for i in range(len(best_route))]
            
            results = {
                'algorithm': 'Genetic Algorithm',
                'best_route': best_route,
                'path_coordinates': path_coordinates,
                'edges': edges,
                'total_distance': final_distance,
                'total_cost': final_cost,
                'initial_distance': initial_distance,
                'initial_cost': initial_cost,
                'improvement_distance': improvement_distance,
                'improvement_cost': improvement_cost,
                'generations': self.generations,
                'cost_history': best_costs,
                'petrol_liters': final_distance / self.petrol_efficiency,
                'execution_time': 0
            }
            
            return edges, path_coordinates, final_distance, best_route, results
            
        except Exception as e:
            return [], [], 0, [], {}

class DFSBruteForce:
    """DFS Brute Force algorithm for TSP (optimal but slow)"""
    
    def __init__(self, max_points=8):
        self.max_points = max_points
        self.petrol_efficiency = 8.0
        self.petrol_cost_per_liter = 1.5
    
    @staticmethod
    def calculate_distance(point1, point2):
        """Calculate Euclidean distance between two points"""
        try:
            return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        except:
            return float('inf')
    
    def calculate_route_distance(self, route, points):
        """Calculate total distance of the route"""
        if len(route) < 2:
            return 0
        
        total_distance = 0
        for i in range(len(route)):
            from_idx = route[i]
            to_idx = route[(i + 1) % len(route)]
            total_distance += self.calculate_distance(points[from_idx], points[to_idx])
        return total_distance
    
    def calculate_petrol_cost(self, route, points):
        """Calculate petrol cost for the route"""
        distance = self.calculate_route_distance(route, points)
        petrol_needed = distance / self.petrol_efficiency
        return petrol_needed * self.petrol_cost_per_liter
    
    def solve_tsp(self, points):
        """Solve TSP using brute force DFS"""
        if len(points) < 2:
            return [], [], 0, [], {}
        
        if len(points) > self.max_points:
            return [], [], 0, [], {'error': f'Too many points for brute force (max {self.max_points}), found {len(points)}'}
        
        try:
            n = len(points)
            best_route = None
            best_distance = float('inf')
            routes_evaluated = 0
            
            # Generate all possible permutations
            for perm in itertools.permutations(range(n)):
                route = list(perm)
                distance = self.calculate_route_distance(route, points)
                routes_evaluated += 1
                
                if distance < best_distance:
                    best_distance = distance
                    best_route = route
            
            final_cost = self.calculate_petrol_cost(best_route, points)
            path_coordinates = [points[i] for i in best_route]
            edges = [(best_route[i], best_route[(i + 1) % len(best_route)]) for i in range(len(best_route))]
            
            results = {
                'algorithm': 'DFS Brute Force',
                'best_route': best_route,
                'path_coordinates': path_coordinates,
                'edges': edges,
                'total_distance': best_distance,
                'total_cost': final_cost,
                'routes_evaluated': routes_evaluated,
                'petrol_liters': best_distance / self.petrol_efficiency,
                'execution_time': 0,
                'optimal': True  # Brute force guarantees optimal solution
            }
            
            return edges, path_coordinates, best_distance, best_route, results
            
        except Exception as e:
            return [], [], 0, [], {'error': str(e)}

class DynamicProgrammingTSP:
    """Dynamic Programming Held-Karp algorithm for TSP (optimal, efficient)"""
    
    def __init__(self, max_points=15):
        self.max_points = max_points
        self.petrol_efficiency = 8.0
        self.petrol_cost_per_liter = 1.5
    
    @staticmethod
    def calculate_distance(point1, point2):
        """Calculate Euclidean distance between two points"""
        try:
            return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        except:
            return float('inf')
    
    def calculate_route_distance(self, route, points):
        """Calculate total distance of the route"""
        if len(route) < 2:
            return 0
        
        total_distance = 0
        for i in range(len(route)):
            from_idx = route[i]
            to_idx = route[(i + 1) % len(route)]
            total_distance += self.calculate_distance(points[from_idx], points[to_idx])
        return total_distance
    
    def calculate_petrol_cost(self, route, points):
        """Calculate petrol cost for the route"""
        distance = self.calculate_route_distance(route, points)
        petrol_needed = distance / self.petrol_efficiency
        return petrol_needed * self.petrol_cost_per_liter
    
    def solve_tsp(self, points):
        """Solve TSP using Held-Karp dynamic programming algorithm"""
        if len(points) < 2:
            return [], [], 0, [], {}
        
        if len(points) > self.max_points:
            return [], [], 0, [], {'error': f'Too many points for DP (max {self.max_points}), found {len(points)}'}
        
        try:
            n = len(points)
            
            # Create distance matrix
            dist = {}
            for i in range(n):
                for j in range(n):
                    dist[(i, j)] = self.calculate_distance(points[i], points[j])
            
            # DP table: memo[mask][i] = (min_cost, parent)
            # mask represents visited nodes, i is current node
            memo = {}
            
            # Base case: starting from node 0, only node 0 visited
            memo[(1, 0)] = (0, -1)
            
            # Fill DP table
            for mask in range(1, 1 << n):
                for u in range(n):
                    if not (mask & (1 << u)):
                        continue
                    
                    if (mask, u) in memo:
                        continue
                    
                    min_cost = float('inf')
                    parent = -1
                    
                    for v in range(n):
                        if u == v or not (mask & (1 << v)):
                            continue
                        
                        prev_mask = mask ^ (1 << u)
                        if (prev_mask, v) in memo:
                            cost = memo[(prev_mask, v)][0] + dist[(v, u)]
                            if cost < min_cost:
                                min_cost = cost
                                parent = v
                    
                    if parent != -1:
                        memo[(mask, u)] = (min_cost, parent)
            
            # Find the minimum cost to visit all nodes and return to start
            final_mask = (1 << n) - 1
            min_cost = float('inf')
            last_node = -1
            
            for i in range(1, n):
                if (final_mask, i) in memo:
                    cost = memo[(final_mask, i)][0] + dist[(i, 0)]
                    if cost < min_cost:
                        min_cost = cost
                        last_node = i
            
            # Reconstruct path
            if last_node == -1:
                return [], [], 0, [], {'error': 'No valid path found'}
            
            path = []
            mask = final_mask
            current = last_node
            
            while current != -1:
                path.append(current)
                if (mask, current) in memo:
                    next_node = memo[(mask, current)][1]
                    mask ^= (1 << current)
                    current = next_node
                else:
                    break
            
            path.reverse()
            
            # Add return to start for closed loop
            if path and path[0] != 0:
                path = [0] + path
            path.append(0)  # Return to start
            
            # Calculate final metrics
            final_distance = self.calculate_route_distance(path, points)
            final_cost = self.calculate_petrol_cost(path, points)
            path_coordinates = [points[i] for i in path]
            edges = [(path[i], path[(i + 1) % len(path)]) for i in range(len(path) - 1)]
            
            # Count subproblems solved
            subproblems_solved = len(memo)
            
            results = {
                'algorithm': 'Dynamic Programming',
                'best_route': path,
                'path_coordinates': path_coordinates,
                'edges': edges,
                'total_distance': final_distance,
                'total_cost': final_cost,
                'subproblems_solved': subproblems_solved,
                'petrol_liters': final_distance / self.petrol_efficiency,
                'execution_time': 0,
                'optimal': True,  # DP guarantees optimal solution
                'space_complexity': f'O(n²×2^n) = O({n}²×2^{n})',
                'time_complexity': f'O(n²×2^n) = O({n}²×2^{n})'
            }
            
            return edges, path_coordinates, final_distance, path, results
            
        except Exception as e:
            return [], [], 0, [], {'error': str(e)}

# Data generation
@st.cache_data
def generate_base_data(num_points=50):
    """Generate base wildfire data without priorities"""
    np.random.seed(42)
    
    damage_types = [
        "destroyed structure", "damaged pipeline", "power line down", 
        "transformer damaged", "vehicle wreckage", "unburned forest",
        "gas leak detected", "water main break", "road blocked",
        "cellular tower down", "damaged bridge", "debris field"
    ]
    
    points = []
    labels = []
    timestamps = []
    
    current_time = datetime.now()
    
    for i in range(num_points):
        if i < num_points // 3:
            x = np.random.normal(-20, 8)
            y = np.random.normal(25, 6)
        elif i < 2 * num_points // 3:
            x = np.random.normal(0, 12)
            y = np.random.normal(0, 10)
        else:
            x = np.random.normal(20, 6)
            y = np.random.normal(-15, 8)
        
        damage = np.random.choice(damage_types)
        timestamp = current_time - timedelta(minutes=random.randint(1, 120))
        
        points.append((x, y))
        labels.append(damage)
        timestamps.append(timestamp)
    
    df = pd.DataFrame({
        'id': range(num_points),
        'longitude': [p[0] for p in points],
        'latitude': [p[1] for p in points],
        'damage_type': labels,
        'timestamp': timestamps,
        'priority': ['Unassigned'] * num_points
    })
    
    return df

def filter_by_single_priority(df, priority):
    """Filter data by a single priority level, excluding unassigned"""
    if priority == 'Unassigned':
        return pd.DataFrame()
    return df[df['priority'] == priority].copy()

def calculate_priority_convex_hulls(df, selected_priorities):
    """Calculate separate convex hulls for each priority level"""
    hull_calc = SafeConvexHull()
    results = {}
    
    for priority in selected_priorities:
        priority_df = filter_by_single_priority(df, priority)
        
        if len(priority_df) >= 3:
            points = list(zip(priority_df['longitude'], priority_df['latitude']))
            hull_points = hull_calc.safe_graham_scan(points)
            
            area, perimeter = 0, 0
            if len(hull_points) >= 3:
                try:
                    from shapely.geometry import Polygon
                    polygon = Polygon(hull_points)
                    area = polygon.area
                    perimeter = polygon.length
                except:
                    pass
            
            results[priority] = {
                'hull_points': hull_points,
                'data_points': points,
                'area': area,
                'perimeter': perimeter,
                'point_count': len(priority_df)
            }
        else:
            results[priority] = {
                'hull_points': [],
                'data_points': [],
                'area': 0,
                'perimeter': 0,
                'point_count': len(priority_df),
                'error': f'Need at least 3 points, found {len(priority_df)}'
            }
    
    return results

def run_all_algorithms(df, selected_priorities):
    """Run all four algorithms and compare results"""
    results = {}
    
    for priority in selected_priorities:
        priority_df = filter_by_single_priority(df, priority)
        
        if len(priority_df) >= 2:
            # Limit points for performance
            max_points_genetic = 20
            max_points_brute = 8
            max_points_dp = 12
            
            if len(priority_df) > max_points_genetic:
                priority_df_genetic = priority_df.sample(n=max_points_genetic, random_state=42)
            else:
                priority_df_genetic = priority_df
            
            if len(priority_df) > max_points_brute:
                priority_df_brute = priority_df.sample(n=max_points_brute, random_state=42)
            else:
                priority_df_brute = priority_df
                
            if len(priority_df) > max_points_dp:
                priority_df_dp = priority_df.sample(n=max_points_dp, random_state=42)
            else:
                priority_df_dp = priority_df
            
            points_genetic = list(zip(priority_df_genetic['longitude'], priority_df_genetic['latitude']))
            points_brute = list(zip(priority_df_brute['longitude'], priority_df_brute['latitude']))
            points_dp = list(zip(priority_df_dp['longitude'], priority_df_dp['latitude']))
            
            # Initialize algorithms
            two_opt = TwoOptHeuristic(max_iterations=1000)
            genetic = GeneticAlgorithmTSP(population_size=100, generations=300)
            brute_force = DFSBruteForce(max_points=8)
            dp_solver = DynamicProgrammingTSP(max_points=12)
            
            algorithm_results = {}
            
            # Run 2-opt Heuristic
            start_time = time.time()
            edges_2opt, path_2opt, dist_2opt, route_2opt, results_2opt = two_opt.solve_tsp(points_genetic)
            results_2opt['execution_time'] = time.time() - start_time
            algorithm_results['2-opt'] = results_2opt
            
            # Run Genetic Algorithm
            start_time = time.time()
            edges_ga, path_ga, dist_ga, route_ga, results_ga = genetic.solve_tsp(points_genetic)
            results_ga['execution_time'] = time.time() - start_time
            algorithm_results['genetic'] = results_ga
            
            # Run Brute Force (if small enough)
            start_time = time.time()
            edges_bf, path_bf, dist_bf, route_bf, results_bf = brute_force.solve_tsp(points_brute)
            results_bf['execution_time'] = time.time() - start_time
            algorithm_results['brute_force'] = results_bf
            
            # Run Dynamic Programming (if small enough)
            start_time = time.time()
            edges_dp, path_dp, dist_dp, route_dp, results_dp = dp_solver.solve_tsp(points_dp)
            results_dp['execution_time'] = time.time() - start_time
            algorithm_results['dynamic_programming'] = results_dp
            
            results[priority] = {
                'algorithms': algorithm_results,
                'point_count_genetic': len(priority_df_genetic),
                'point_count_brute': len(priority_df_brute),
                'point_count_dp': len(priority_df_dp),
                'data_genetic': priority_df_genetic,
                'data_brute': priority_df_brute,
                'data_dp': priority_df_dp
            }
        else:
            results[priority] = {
                'algorithms': {},
                'point_count_genetic': len(priority_df),
                'point_count_brute': len(priority_df),
                'point_count_dp': len(priority_df),
                'data_genetic': priority_df,
                'data_brute': priority_df,
                'data_dp': priority_df,
                'error': f'Need at least 2 points, found {len(priority_df)}'
            }
    
    return results

# Main application
def main():
    st.markdown('<h1 class="main-header">🔥 Advanced Algorithm Comparison System</h1>', unsafe_allow_html=True)
    
    init_session_state()
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="algorithm-comparison">System calculating area under a region and calculating optimal travelling path</div>', unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.title("🎛️ Advanced Algorithm Controls")
        
        st.subheader("📊 Data Generation")
        num_points = st.slider("Number of Data Points", 8, 50, 100, 200)
        
        if st.button("🔄 **REFRESH DATA**", type="primary", use_container_width=True):
            st.session_state.base_data = generate_base_data(num_points)
            st.session_state.data_loaded = True
            st.session_state.priorities_set = False
            st.session_state.convex_hull_calculated = False
            st.session_state.algorithms_calculated = False
            st.session_state.priority_results = {'convex_hulls': {}, 'algorithm_results': {}}
            st.session_state.last_manual_refresh = datetime.now()
            st.rerun()
        
        if st.session_state.data_loaded:
            st.info(f"Last refresh: {st.session_state.last_manual_refresh.strftime('%H:%M:%S')}")
        
        st.markdown("---")
        
        st.subheader("🎯 Algorithm Analysis")
        
        st.session_state.selected_priorities_for_analysis = st.multiselect(
            "Select priorities for comparison:",
            ['Critical', 'Medium', 'Low'],
            default=['Critical'],
            key="priority_selector",
            help="Compare all 4 algorithms across priority levels"
        )
        
        # Algorithm information
        st.markdown("### 🧠 Algorithm Overview")
        st.markdown("""
        - **2-opt**: Fast local optimization
        - **Genetic**: Population-based evolution
        - **DFS Brute**: Exhaustive search (≤8 pts)
        - **Dynamic Programming**: Held-Karp optimal (≤12 pts)
        """)
        
        if st.button("🔺 **CALCULATE CONVEX HULLS**", type="secondary", use_container_width=True):
            if st.session_state.priorities_set:
                st.session_state.convex_hull_calculated = True
                st.rerun()
            else:
                st.error("Please set priorities first!")
        
        if st.button("🧠 **RUN ALL 4 ALGORITHMS**", type="secondary", use_container_width=True):
            if st.session_state.priorities_set:
                st.session_state.algorithms_calculated = True
                st.rerun()
            else:
                st.error("Please set priorities first!")
        
        if st.button("🔄 **RESET ALL**", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_session_state()
            st.rerun()
    
    if not st.session_state.data_loaded:
        st.info("👆 Click **REFRESH DATA** in the sidebar to load incident data")
        return
    
    df = st.session_state.base_data
    
    tab1, tab2, tab3 = st.tabs(["📊 Data & Priority Setting", "🔺 Convex Hull Analysis", "🧠 Advanced Algorithm Comparison"])
    
    with tab1:
        show_data_and_priority_setting(df)
    
    with tab2:
        show_multi_priority_convex_hull_analysis(df)
    
    with tab3:
        show_advanced_algorithm_comparison_analysis(df)

def show_data_and_priority_setting(df):
    """Show data and priority setting interface - ALL THREE METHODS"""
    st.subheader("📊 Incident Data & Manual Priority Assignment")
    
    # Enhanced stats with assignment tracking
    col1, col2, col3, col4 = st.columns(4)
    
    total_incidents = len(df)
    assigned_count = len(df[df['priority'] != 'Unassigned'])
    unassigned_count = len(df[df['priority'] == 'Unassigned'])
    critical_count = len(df[df['priority'] == 'Critical'])
    
    with col1:
        st.metric("Total Incidents", total_incidents)
    with col2:
        st.metric("Assigned Priorities", assigned_count, delta=f"-{unassigned_count} unassigned")
    with col3:
        st.metric("Critical Assigned", critical_count)
    with col4:
        completion = (assigned_count / total_incidents) * 100 if total_incidents > 0 else 0
        st.metric("Assignment Progress", f"{completion:.1f}%")
    
    # Priority breakdown
    priority_breakdown = df['priority'].value_counts()
    col1, col2 = st.columns(2)
    
    with col1:
        # Show assignment status
        if unassigned_count > 0:
            st.warning(f"⚠️ {unassigned_count} incidents still need priority assignment. These will be excluded from analysis.")
        else:
            st.success("✅ All incidents have assigned priorities!")
    
    with col2:
        # Priority distribution chart
        if len(priority_breakdown) > 0:
            color_map = {'Critical': 'red', 'Medium': 'orange', 'Low': 'green', 'Unassigned': 'gray'}
            fig_pie = px.pie(
                values=priority_breakdown.values,
                names=priority_breakdown.index,
                title="Current Priority Distribution",
                color_discrete_map=color_map
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # Priority assignment method selection - ALL THREE METHODS
    st.subheader("🎯 Priority Assignment Method")
    
    priority_method = st.radio(
        "Choose how to assign priorities:",
        ["Manual Individual Assignment", "Bulk Assignment by Damage Type", "Quick Priority Assignment"],
        horizontal=True
    )
    
    if priority_method == "Manual Individual Assignment":
        show_manual_individual_assignment(df)
    elif priority_method == "Bulk Assignment by Damage Type":
        show_bulk_assignment(df)
    else:
        show_quick_assignment(df)

def show_manual_individual_assignment(df):
    """Show manual individual priority assignment"""
    st.subheader("🔧 Individual Priority Assignment")
    
    # Create editable dataframe
    edited_df = st.data_editor(
        df,
        column_config={
            "id": st.column_config.NumberColumn("ID", disabled=True),
            "longitude": st.column_config.NumberColumn("Longitude", format="%.2f", disabled=True),
            "latitude": st.column_config.NumberColumn("Latitude", format="%.2f", disabled=True),
            "damage_type": st.column_config.TextColumn("Damage Type", disabled=True),
            "timestamp": st.column_config.DatetimeColumn("Timestamp", disabled=True),
            "priority": st.column_config.SelectboxColumn(
                "Priority",
                options=["Unassigned", "Critical", "Medium", "Low"],
                default="Unassigned"
            )
        },
        hide_index=True,
        use_container_width=True,
        height=400
    )
    
    # Save priorities button
    if st.button("💾 **SAVE PRIORITIES**", type="primary"):
        st.session_state.base_data = edited_df
        st.session_state.priorities_set = True
        st.success("✅ Priorities saved successfully!")
        st.rerun()

def show_bulk_assignment(df):
    """Show bulk priority assignment by damage type"""
    st.subheader("📦 Bulk Assignment by Damage Type")
    
    # Get unique damage types
    damage_types = df['damage_type'].unique()
    
    # Create priority mapping
    priority_mapping = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Assign priorities to damage types:**")
        for damage_type in damage_types:
            priority_mapping[damage_type] = st.selectbox(
                f"{damage_type}",
                ["Unassigned", "Critical", "Medium", "Low"],
                key=f"bulk_{damage_type}"
            )
    
    with col2:
        st.write("**Preview of assignments:**")
        for damage_type, priority in priority_mapping.items():
            count = len(df[df['damage_type'] == damage_type])
            color_class = f"priority-{priority.lower()}" if priority != "Unassigned" else ""
            st.markdown(f"• {damage_type}: <span class='{color_class}'>{priority}</span> ({count} incidents)", unsafe_allow_html=True)
    
    # Apply bulk assignment
    if st.button("🔄 **APPLY BULK ASSIGNMENT**", type="primary"):
        df_updated = df.copy()
        for damage_type, priority in priority_mapping.items():
            df_updated.loc[df_updated['damage_type'] == damage_type, 'priority'] = priority
        
        st.session_state.base_data = df_updated
        st.session_state.priorities_set = True
        st.success("✅ Bulk assignment applied successfully!")
        st.rerun()

def show_quick_assignment(df):
    """Show quick priority assignment interface"""
    st.subheader("⚡ Quick Priority Assignment")
    
    # Interactive map for quick assignment
    st.write("**Use the map to visualize and assign priorities:**")
    
    # Create interactive map
    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color="priority",
        size_max=15,
        hover_data={'damage_type': True, 'id': True},
        color_discrete_map={
            'Unassigned': 'gray',
            'Critical': 'red',
            'Medium': 'orange',
            'Low': 'green'
        },
        mapbox_style="open-street-map",
        height=500,
        title="Current Priority Assignment Status"
    )
    
    fig.update_layout(
        mapbox=dict(
            center=dict(lat=df['latitude'].mean(), lon=df['longitude'].mean()),
            zoom=8
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Quick assignment controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_ids = st.multiselect(
            "Select incident IDs:",
            df['id'].tolist(),
            key="quick_select_ids"
        )
    
    with col2:
        quick_priority = st.selectbox(
            "Assign priority:",
            ["Critical", "Medium", "Low"],
            key="quick_priority"
        )
    
    with col3:
        if st.button("✅ **ASSIGN SELECTED**", type="primary"):
            if selected_ids:
                df_updated = df.copy()
                df_updated.loc[df_updated['id'].isin(selected_ids), 'priority'] = quick_priority
                st.session_state.base_data = df_updated
                st.session_state.priorities_set = True
                st.success(f"✅ Assigned {quick_priority} priority to {len(selected_ids)} incidents!")
                st.rerun()
            else:
                st.warning("Please select at least one incident ID")

def show_multi_priority_convex_hull_analysis(df):
    """Show convex hull analysis"""
    st.subheader("🔺 Multi-Priority Convex Hull Analysis")
    
    if not st.session_state.priorities_set:
        st.warning("⚠️ Please set priorities in the first tab")
        return
    
    if not st.session_state.convex_hull_calculated:
        st.info("👆 Click **CALCULATE CONVEX HULLS** in the sidebar")
        return
    
    selected_priorities = st.session_state.selected_priorities_for_analysis
    hull_results = calculate_priority_convex_hulls(df, selected_priorities)
    
    color_map = {'Critical': '#f44336', 'Medium': '#ff9800', 'Low': '#4caf50'}
    
    for priority in selected_priorities:
        priority_class = f"priority-{priority.lower()}"
        st.markdown(f'<div class="priority-section {priority_class}">', unsafe_allow_html=True)
        st.subheader(f"🔺 {priority} Priority Zone Analysis")
        
        result = hull_results[priority]
        
        if 'error' in result:
            st.warning(f"❌ {result['error']}")
        else:
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Zone Vertices", len(result['hull_points']))
            with col2:
                st.metric("Points in Zone", result['point_count'])
            with col3:
                st.metric("Zone Area", f"{result['area']:.2f} sq units")
            with col4:
                st.metric("Zone Perimeter", f"{result['perimeter']:.2f} units")
            
            # Visualization for this priority
            fig = go.Figure()
            
            # Add data points for this priority
            priority_data = df[df['priority'] == priority]
            if len(priority_data) > 0:
                fig.add_trace(go.Scatter(
                    x=priority_data['longitude'],
                    y=priority_data['latitude'],
                    mode='markers',
                    marker=dict(color=color_map[priority], size=10),
                    name=f'{priority} Priority Points',
                    hovertemplate='<b>%{text}</b><br>Priority: %{customdata}<extra></extra>',
                    text=priority_data['damage_type'],
                    customdata=priority_data['priority']
                ))
            
            # Add convex hull
            if result['hull_points']:
                hull_x = [p[0] for p in result['hull_points']] + [result['hull_points'][0][0]]
                hull_y = [p[1] for p in result['hull_points']] + [result['hull_points'][0][1]]
                
                fig.add_trace(go.Scatter(
                    x=hull_x,
                    y=hull_y,
                    mode='lines',
                    line=dict(color=color_map[priority], width=3),
                    name=f'{priority} Zone Boundary',
                    fill='toself',
                    fillcolor=f'rgba({",".join(str(int(color_map[priority][i:i+2], 16)) for i in (1, 3, 5))}, 0.2)'
                ))
            
            fig.update_layout(
                title=f"{priority} Priority Zone",
                xaxis_title="Longitude",
                yaxis_title="Latitude",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Export option for this priority
            if result['hull_points']:
                hull_df = pd.DataFrame(result['hull_points'], columns=['Longitude', 'Latitude'])
                csv = hull_df.to_csv(index=False)
                st.download_button(
                    f"📥 Download {priority} Zone Coordinates",
                    csv,
                    f"{priority.lower()}_zone_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    key=f"download_hull_{priority}"
                )
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_advanced_algorithm_comparison_analysis(df):
    """Show comprehensive algorithm comparison with Dynamic Programming"""
    st.subheader("🧠 Advanced Algorithm Comparison Analysis")
    
    if not st.session_state.priorities_set:
        st.warning("⚠️ Please set priorities in the first tab")
        return
    
    if not st.session_state.algorithms_calculated:
        st.info("👆 Click **RUN ALL 4 ALGORITHMS** in the sidebar to compare optimization methods")
        return
    
    selected_priorities = st.session_state.selected_priorities_for_analysis
    
    with st.spinner("🧠 Running comprehensive algorithm comparison..."):
        algorithm_results = run_all_algorithms(df, selected_priorities)
    
    st.session_state.priority_results['algorithm_results'] = algorithm_results
    
    # Algorithm performance summary
    st.subheader("📊 Advanced Algorithm Performance Summary")
    
    summary_data = []
    for priority in selected_priorities:
        if 'error' not in algorithm_results[priority]:
            algorithms = algorithm_results[priority]['algorithms']
            
            for algo_name, results in algorithms.items():
                if 'error' not in results:
                    summary_data.append({
                        'Priority': priority,
                        'Algorithm': results.get('algorithm', algo_name),
                        'Distance (km)': f"{results.get('total_distance', 0):.2f}",
                        'Cost ($)': f"{results.get('total_cost', 0):.2f}",
                        'Petrol (L)': f"{results.get('petrol_liters', 0):.2f}",
                        'Time (s)': f"{results.get('execution_time', 0):.4f}",
                        'Optimal': '✅' if results.get('optimal', False) else '❌',
                        'Points': results.get('point_count_genetic', 0) if 'genetic' in algo_name.lower() else results.get('point_count_brute', 0) if 'brute' in algo_name.lower() else results.get('point_count_dp', 0)
                    })
    
    if summary_data:
        comparison_df = pd.DataFrame(summary_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Cost comparison chart
        fig_comparison = px.bar(
            comparison_df,
            x='Algorithm',
            y='Cost ($)',
            color='Priority',
            title="Petrol Cost Comparison - All 4 Algorithms",
            barmode='group'
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Execution time comparison
        fig_time = px.bar(
            comparison_df,
            x='Algorithm',
            y='Time (s)',
            color='Priority',
            title="Execution Time Comparison - All 4 Algorithms",
            barmode='group'
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    # Detailed analysis for each priority
    color_map = {'Critical': '#f44336', 'Medium': '#ff9800', 'Low': '#4caf50'}
    
    for priority in selected_priorities:
        if 'error' not in algorithm_results[priority]:
            st.markdown(f'<div class="priority-section priority-{priority.lower()}">', unsafe_allow_html=True)
            st.subheader(f"🧠 {priority} Priority - Advanced Algorithm Comparison")
            
            algorithms = algorithm_results[priority]['algorithms']
            
            # Create comparison cards for all 4 algorithms
            col1, col2, col3, col4 = st.columns(4)
            
            # 2-opt Heuristic
            with col1:
                if '2-opt' in algorithms and 'error' not in algorithms['2-opt']:
                    result = algorithms['2-opt']
                    st.markdown('<div class="algorithm-card heuristic-card">', unsafe_allow_html=True)
                    st.subheader("🔄 2-opt Heuristic")
                    st.metric("Cost", f"${result.get('total_cost', 0):.2f}")
                    st.metric("Distance", f"{result.get('total_distance', 0):.2f} km")
                    st.metric("Time", f"{result.get('execution_time', 0):.4f}s")
                    st.metric("Improvements", result.get('improvements', 0))
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Genetic Algorithm
            with col2:
                if 'genetic' in algorithms and 'error' not in algorithms['genetic']:
                    result = algorithms['genetic']
                    st.markdown('<div class="algorithm-card genetic-card">', unsafe_allow_html=True)
                    st.subheader("🧬 Genetic Algorithm")
                    st.metric("Cost", f"${result.get('total_cost', 0):.2f}")
                    st.metric("Distance", f"{result.get('total_distance', 0):.2f} km")
                    st.metric("Time", f"{result.get('execution_time', 0):.4f}s")
                    st.metric("Generations", result.get('generations', 0))
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Brute Force
            with col3:
                if 'brute_force' in algorithms:
                    result = algorithms['brute_force']
                    st.markdown('<div class="algorithm-card brute-force-card">', unsafe_allow_html=True)
                    st.subheader("🔍 DFS Brute Force")
                    if 'error' not in result:
                        st.metric("Cost", f"${result.get('total_cost', 0):.2f}")
                        st.metric("Distance", f"{result.get('total_distance', 0):.2f} km")
                        st.metric("Time", f"{result.get('execution_time', 0):.4f}s")
                        st.metric("Routes Checked", result.get('routes_evaluated', 0))
                        if result.get('optimal'):
                            st.success("✅ Optimal Solution")
                    else:
                        st.warning(f"❌ {result['error']}")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Dynamic Programming
            with col4:
                if 'dynamic_programming' in algorithms:
                    result = algorithms['dynamic_programming']
                    st.markdown('<div class="algorithm-card dp-card">', unsafe_allow_html=True)
                    st.subheader("⚡ Dynamic Programming")
                    if 'error' not in result:
                        st.metric("Cost", f"${result.get('total_cost', 0):.2f}")
                        st.metric("Distance", f"{result.get('total_distance', 0):.2f} km")
                        st.metric("Time", f"{result.get('execution_time', 0):.4f}s")
                        st.metric("Subproblems", f"{result.get('subproblems_solved', 0):,}")
                        if result.get('optimal'):
                            st.success("✅ Optimal Solution")
                        st.info(f"🛣️ {result.get('path_type', 'Open Path')}")
                    else:
                        st.warning(f"❌ {result['error']}")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Visualization comparison
            st.subheader(f"🗺️ {priority} Priority Route Comparison - All 4 Algorithms")
            
            # Create subplots for route comparison
            valid_algorithms = []
            
            for algo_name, result in algorithms.items():
                if 'error' not in result and result.get('path_coordinates'):
                    valid_algorithms.append((algo_name, result))
            
            if valid_algorithms:
                # Determine number of columns based on valid algorithms
                num_cols = min(4, len(valid_algorithms))
                if num_cols == 3:
                    cols = st.columns([1, 1, 1])
                elif num_cols == 4:
                    cols = st.columns(4)
                else:
                    cols = st.columns(num_cols)
                
                for i, (algo_name, result) in enumerate(valid_algorithms):
                    with cols[i % len(cols)]:
                        fig = go.Figure()
                        
                        path_x = [coord[0] for coord in result['path_coordinates']]
                        path_y = [coord[1] for coord in result['path_coordinates']]
                        
                        # Route path
                        fig.add_trace(go.Scatter(
                            x=path_x,
                            y=path_y,
                            mode='lines+markers',
                            line=dict(color=color_map[priority], width=3),
                            marker=dict(color=color_map[priority], size=8),
                            name=f'{result.get("algorithm", algo_name)}'
                        ))
                        
                        # Start point
                        fig.add_trace(go.Scatter(
                            x=[path_x[0]],
                            y=[path_y[0]],
                            mode='markers',
                            marker=dict(color='gold', size=15, symbol='star'),
                            name='Start'
                        ))
                        
                        # End point (different for DP open path)
                        if result.get('path_type') == 'Open Path':
                            fig.add_trace(go.Scatter(
                                x=[path_x[-1]],
                                y=[path_y[-1]],
                                mode='markers',
                                marker=dict(color='darkred', size=15, symbol='square'),
                                name='End'
                            ))
                        
                        title_text = f"{result.get('algorithm', algo_name)}<br>Cost: ${result.get('total_cost', 0):.2f}"
                        if result.get('optimal'):
                            title_text += "<br>✅ Optimal"
                        
                        fig.update_layout(
                            title=title_text,
                            height=300,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Overall comparison summary with 4 algorithms
    if summary_data:
        st.subheader("🏆 Advanced Algorithm Ranking Summary")
        
        # Calculate best algorithms by different criteria
        comparison_df = pd.DataFrame(summary_data)
        comparison_df['Cost_Numeric'] = comparison_df['Cost ($)'].str.replace('$', '').astype(float)
        comparison_df['Time_Numeric'] = comparison_df['Time (s)'].str.replace('s', '').astype(float)
        
        best_cost = comparison_df.loc[comparison_df['Cost_Numeric'].idxmin()]
        fastest = comparison_df.loc[comparison_df['Time_Numeric'].idxmin()]
        optimal_algorithms = comparison_df[comparison_df['Optimal'] == '✅']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.success(f"🏆 **Best Cost**: {best_cost['Algorithm']} (${best_cost['Cost_Numeric']:.2f})")
        
        with col2:
            st.success(f"⚡ **Fastest**: {fastest['Algorithm']} ({fastest['Time_Numeric']:.4f}s)")
        
        with col3:
            if len(optimal_algorithms) > 0:
                st.success(f"✅ **Optimal Solutions**: {len(optimal_algorithms)} algorithms")
            else:
                st.info("ℹ️ **No optimal solutions** for current dataset size")
        
        with col4:
            # Calculate efficiency (cost vs time trade-off)
            comparison_df['Efficiency'] = 1 / (comparison_df['Cost_Numeric'] * comparison_df['Time_Numeric'])
            most_efficient = comparison_df.loc[comparison_df['Efficiency'].idxmax()]
            st.success(f"⚖️ **Most Efficient**: {most_efficient['Algorithm']}")
        
        # Algorithm recommendations
        st.subheader("🎯 Algorithm Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🚀 **For Speed:**")
            st.markdown("- **2-opt Heuristic**: Fast local optimization")
            st.markdown("- **Genetic Algorithm**: Good balance of speed and quality")
            
            st.markdown("### 🎯 **For Accuracy:**")
            st.markdown("- **Dynamic Programming**: Optimal solutions up to 12 points")
            st.markdown("- **DFS Brute Force**: Optimal solutions up to 8 points")
        
        with col2:
            st.markdown("### 📊 **Use Cases:**")
            st.markdown("- **Real-time Operations**: Use 2-opt or Genetic")
            st.markdown("- **Critical Missions**: Use DP or Brute Force")
            st.markdown("- **Large Datasets**: Use 2-opt or Genetic only")
            st.markdown("- **Small High-Value Operations**: Use DP for optimal results")

if __name__ == "__main__":
    main()
