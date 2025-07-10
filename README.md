# 🔥 Wildfire Emergency Response System

An advanced algorithm comparison system for optimizing emergency response routes during wildfire incidents using multiple Traveling Salesman Problem (TSP) algorithms.
## Some live working photos
<img width="3588" height="1988" alt="image" src="https://github.com/user-attachments/assets/3919ce2e-8b1c-47a1-8138-2c576d26d213" />
<img width="3568" height="2006" alt="image" src="https://github.com/user-attachments/assets/ce3d1950-ac1d-4e82-ac74-b996919ab5d4" />
<img width="3600" height="2000" alt="image" src="https://github.com/user-attachments/assets/be27b30d-52db-4eea-abad-0db1413da94d" />
<img width="3600" height="1996" alt="image" src="https://github.com/user-attachments/assets/9d0255c1-c69b-431d-ad45-6102feacc491" />

## Link to live working
https://www.youtube.com/watch?v=5ITo3Sy_S7o




## 🌟 Overview

This Streamlit web application provides a comprehensive platform for comparing different TSP algorithms in the context of wildfire emergency response optimization. The system helps emergency response teams find the most efficient routes to visit multiple incident locations while considering factors like fuel efficiency, costs, and response priorities.

## 🚀 Key Features

### 🔧 Multi-Algorithm TSP Solver
- **2-opt Heuristic**: Fast local search optimization
- **Genetic Algorithm**: Evolutionary approach with population-based search
- **DFS Brute Force**: Exhaustive search for guaranteed optimal solutions
- **Dynamic Programming**: Held-Karp algorithm for optimal TSP solutions

### 📊 Advanced Analytics
- Real-time performance metrics comparison
- Route cost calculations with petrol efficiency modeling
- Priority-based incident categorization (Critical, Medium, Low)
- Convex hull visualization for affected areas
- Distance and cost optimization tracking

### 🎯 Emergency Response Features
- Wildfire incident data simulation
- Priority-based route optimization
- Multi-criteria decision support
- Interactive data visualization
- Real-time algorithm performance analysis

## 🛠️ Technical Specifications

### Algorithm Characteristics

| Algorithm | Time Complexity | Space Complexity | Max Points | Use Case |
|-----------|----------------|------------------|------------|----------|
| 2-opt Heuristic | O(n²) per iteration | O(n) | Unlimited | Fast approximate solutions |
| Genetic Algorithm | O(g×p×n²) | O(p×n) | 20 | Balanced quality and speed |
| DFS Brute Force | O(n!) | O(n) | 8 | Guaranteed optimal (small datasets) |
| Dynamic Programming | O(n²×2ⁿ) | O(n×2ⁿ) | 15 | Optimal solution (medium datasets) |

*Where: n = number of points, g = generations, p = population size*

### System Parameters
- **Petrol Efficiency**: 8.0 km per liter
- **Petrol Cost**: 1.5 currency units per liter
- **Max Data Points**: 50 (configurable)
- **Priority Levels**: Critical, Medium, Low
- **Damage Types**: 12 different incident categories

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/wildfire-emergency-response.git

# Navigate to the project directory
cd wildfire-emergency-response

# Install required dependencies
pip install streamlit pandas numpy matplotlib plotly networkx shapely

# Run the application
streamlit run wildfire_app.py
```

## 🔧 Dependencies

```python
streamlit
pandas
numpy
matplotlib
plotly
networkx
shapely
```

## 📈 Usage

### 1. Launch the Application
```bash
streamlit run wildfire_app.py
```

### 2. Data Generation
- The system automatically generates synthetic wildfire incident data
- Includes geographic coordinates, damage types, and timestamps
- Supports up to 50 incident points with configurable clustering

### 3. Priority Assignment
- Classify incidents into Critical, Medium, or Low priority
- Filter and analyze data by priority levels
- Generate separate optimization routes for each priority

### 4. Algorithm Comparison
- Run multiple TSP algorithms simultaneously
- Compare performance metrics including:
  - Total distance
  - Fuel cost
  - Execution time
  - Route quality
  - Optimization improvements

### 5. Visualization
- Interactive maps showing incident locations
- Optimized route visualization
- Convex hull boundaries for affected areas
- Real-time performance charts

## 🧮 Algorithm Details

### 2-opt Heuristic
- **Description**: Local search algorithm that iteratively improves routes
- **Advantages**: Fast execution, good for large datasets
- **Limitations**: May find local optima, not guaranteed optimal

### Genetic Algorithm
- **Description**: Evolutionary approach using selection, crossover, and mutation
- **Parameters**:
  - Population Size: 100
  - Elite Size: 20
  - Mutation Rate: 0.01
  - Generations: 300
- **Advantages**: Good balance of quality and performance
- **Limitations**: Stochastic results, parameter tuning required

### DFS Brute Force
- **Description**: Exhaustive search of all possible routes
- **Advantages**: Guaranteed optimal solution
- **Limitations**: Exponential time complexity, limited to ~8 points

### Dynamic Programming (Held-Karp)
- **Description**: Optimal TSP solution using dynamic programming
- **Advantages**: Guaranteed optimal, better than brute force
- **Limitations**: Exponential space complexity, limited to ~15 points

## 🎯 Use Cases

### Emergency Response Planning
- Optimize routes for fire trucks and emergency vehicles
- Minimize response time and fuel consumption
- Coordinate multi-team deployment strategies

### Resource Allocation
- Efficient distribution of emergency supplies
- Optimal positioning of mobile command centers
- Strategic placement of temporary facilities

### Risk Assessment
- Analyze affected area boundaries using convex hulls
- Prioritize incidents based on severity and accessibility
- Support decision-making with quantitative metrics

## 🔍 Code Structure

```
wildfire_app.py
├── 🏗️ Configuration & Setup
├── 🔧 Algorithm Classes
│   ├── SafeStringMatcher
│   ├── SafeConvexHull
│   ├── TwoOptHeuristic
│   ├── GeneticAlgorithmTSP
│   ├── DFSBruteForce
│   └── DynamicProgrammingTSP
├── 📊 Data Generation
├── 🎯 Priority Management
├── 🔄 Algorithm Execution
└── 🎨 Visualization & UI
```

## 📊 Performance Metrics

The system tracks and compares:
- **Distance Optimization**: Total route distance minimization
- **Cost Efficiency**: Fuel consumption and operational costs
- **Execution Time**: Algorithm performance benchmarking
- **Solution Quality**: Improvement percentages over initial routes
- **Scalability**: Performance with varying dataset sizes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Streamlit for the excellent web framework
- The TSP algorithm research community
- Emergency response organizations for inspiring this work

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue or reach out via email.

---

*Built with ❤️ for emergency response optimization*
