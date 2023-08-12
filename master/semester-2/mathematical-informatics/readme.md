# Mathematical informatics

[GO BACK](https://github.com/0xMartin/UTB-FAI-programs)

## Task 1
The code defines several mathematical functions such as Sphere, Rosenbrock, Rastrigin, Ackley, and Booth Function, as well as a Random Search algorithm to find their minimum values.

The numpy and matplotlib.pyplot libraries are imported for numerical and graphical computations. The sphere(), rosenbrock(), rastrigin(), ackley(), and boothFunction() functions take a list of numerical values as input and return the value of the corresponding mathematical function.

The randomSearch() function takes the function to optimize, dimensionality, and maximum number of function evaluations as input and returns the best solution, the function value at the best solution, and a list of function values at each iteration.

The plotFunction() function visualizes a mathematical function as either a 2D or 3D plot. It takes the function, a boolean value to indicate whether to plot in 3D, title of the plot, and the range of values for two parameters as input.

In the main section of the code, the plotFunction() function is called to visualize each of the test functions, and the randomSearch() function is called to find the minimum of the Sphere function for 5, 10, and 20 dimensions. Finally, a convergence plot is shown for each of these calls to randomSearch().

<div>
    <img src="./img/task-1/img1.png" width="48%">
    <img src="./img/task-1/img2.png" width="48%">
</div>
<img src="./img/task-1/img3.png">

## Task 2
This is a Python script that defines several test functions commonly used in optimization problems, and two optimization algorithms: Stochastic Hill Climber and Local Search.

The test functions defined in this script are: sphere, rosenbrock, rastrigin, ackley, and boothFunction. These functions take a list of variables and return a scalar value.

The Stochastic Hill Climber (SHC) algorithm and the Local Search (LS) algorithm are also defined in the script. These algorithms both take a function, the dimensionality of the search space, the minimum and maximum values for the search space, the maximum number of iterations, the size of the neighborhood, and the number of neighbors to evaluate. They both return the best found solution and the history of the algorithm.

Additionally, there is a helper function called plotFunctionWithPts that visualizes the history of the optimization algorithms by plotting the function and the points where the algorithms found the best solution. This function takes the function, the history of the algorithm, the best found solution, and the minimum and maximum values for the search space.

The numpy and matplotlib.pyplot libraries are imported at the beginning of the script.

<div>
    <img src="./img/task-2/img1.png" width="48%">
    <img src="./img/task-2/img2.png" width="48%">
</div>

## Task 3
This code implements three heuristics for function optimization: Stochastic Hill Climber (SHC), Local Search (LS), and Simulated Annealing (SA). All three heuristics share input parameters such as the dimension size (dim), the minimum and maximum parameter values (x_min, x_max), and the maximum number of iterations (max_iter).

SHC and LS differ in how they select neighbors and how they move in the solution space. SHC selects a neighbor randomly and moves in a random direction. On the other hand, LS selects a certain number of neighbors randomly and chooses the best neighbor to improve the current solution.

SA differs in that it works with temperature, which gradually decreases and affects the probability of accepting a worse solution. SA also selects a neighbor randomly and moves in a random direction in the solution space.

The output of each heuristic is the best found parameter values and the value of the objective function, as well as the history of the best objective function values during iterations.

<div>
    <img src="./img/task-3/img1.png" width="48%">
    <img src="./img/task-3/img2.png" width="48%">
</div>

## Task 4

This task contains an implementation aimed at solving the knapsack problem with a unique twist. In this problem, a set of items is generated, each belonging to a specific category. Each item within a category is identified by its ID and possesses distinct weight and value attributes. The objective is to select a combination of items for the knapsack such that only one item is chosen from each category, the total weight does not exceed the maximum capacity of the knapsack, and the total value is maximized.

#### Items (5 classes)
<img src="./img/task-4/img1.png" width="60%">

#### Best combination (for list with 12 classes)
<img src="./img/task-4/img2.png" width="60%">

#### Conclusion
In one hour, it is possible to solve a problem with 12 categories (3 items per category) on my PC. The estimated time for solving the 12-category problem is 18.07 minutes. The actual solution time was 1422.04 seconds (23.7 minutes). For solving a problem with 13 categories, it would require approximately 2 hours.

## Task 5

This task serves as a comparative study between two methods for solving the knapsack problem: the brute force approach and the Hill Climbing (HC) heuristic method. The primary goal is to analyze and contrast the performance of these two techniques in solving knapsack problems efficiently.

### Methods Compared

1. __Brute Force Method:__ This involves systematically considering all possible combinations of items and evaluating their total value while ensuring they do not exceed the knapsack's capacity. While effective for small instances, this method becomes impractical as the problem size increases due to its exponential time complexity.

1. __Hill Climbing Heuristic:__ The Hill Climbing algorithm employs a local search strategy to iteratively improve the solution. Starting from a random or initial solution, it explores neighboring solutions and moves towards the one with higher value, iteratively refining the result. Hill Climbing is often faster than brute force for larger problem instances.

<div>
    <img src="./img/task-5/img1.png" width="48%">
    <img src="./img/task-5/img2.png" width="48%">
</div>

### Conclusion
The comparative study between the brute force and Hill Climbing heuristic methods aims to highlight the trade-off between accuracy and computational efficiency. While the brute force method guarantees an optimal solution, the Hill Climbing heuristic offers a significantly faster approach, making it a preferable choice for larger instances of the knapsack problem. 