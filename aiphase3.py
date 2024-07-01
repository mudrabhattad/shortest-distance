import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from queue import PriorityQueue
import random

class Node:
    def __init__(self, City, D_City, actual_distance_to_destination, actual_time, priority_score):
        self.City = City
        self.D_City = D_City
        self.actual_distance_to_destination = actual_distance_to_destination
        self.actual_time = actual_time
        self.priority_score = priority_score

    def __lt__(self, other):
        return self.priority_score < other.priority_score

def heuristic(node):
    distance_weight = 0.3
    time_weight = 0.3
    priority_weight = 0.2
    estimated_cost = (distance_weight * node.actual_distance_to_destination) + \
                     (time_weight * node.actual_time) + \
                     (priority_weight * node.priority_score)
    return estimated_cost

def best_first_search(start_node, goal_node, heuristic, df):
    frontier = PriorityQueue()
    frontier.put((0, start_node))
    came_from = {}
    cost_so_far = {}
    came_from[start_node.City] = None
    cost_so_far[start_node.City] = 0
    all_paths = []

    while not frontier.empty():
        current_cost, current_node = frontier.get()
        if current_node.City == goal_node.City:
            path = []
            while current_node != start_node:
                path.append(current_node)
                current_node = came_from[current_node.City]
            path.append(start_node)
            path.reverse()
            all_paths.append(path)
            continue
        for next_node in get_neighbors(current_node, df):
            new_cost = cost_so_far[current_node.City] + get_cost(current_node, next_node)
            if next_node.City not in cost_so_far or new_cost < cost_so_far[next_node.City]:
                cost_so_far[next_node.City] = new_cost
                priority = new_cost + heuristic(next_node)
                frontier.put((priority, next_node))
                came_from[next_node.City] = current_node
    return all_paths

def get_neighbors(node, df):
    neighbors = []
    for _, row in df.iterrows():
        if row['City'] == node.City and row['D_City'] != node.D_City:
            neighbor_node = Node(row['D_City'], None, row['actual_distance_to_destination'], row['actual_time'], row['priority_score'])
            neighbors.append(neighbor_node)
    return neighbors

def get_cost(node1, node2):
    return node2.actual_distance_to_destination

def generate_fake_routes(source, destination, num_fake_routes=8):
    fake_routes = []
    intermediary_cities = set(df['City']).difference({source, destination})
    
    for _ in range(num_fake_routes):
        fake_route = [source]  # Start with the source city
        
        # Choose 10 intermediary cities
        for _ in range(10):
            intermediary_city = random.choice(list(intermediary_cities))
            fake_route.append(intermediary_city)
            intermediary_cities.remove(intermediary_city)
        
        fake_route.append(destination)  # End with the destination city
        fake_routes.append(fake_route)
    
    return fake_routes



def plot_graph(df, optimal_path, min_cost, fake_routes):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['City'], row['D_City'], weight=row['actual_distance_to_destination'])

    plt.figure(figsize=(10, 6))

    # Get all nodes in the optimal path
    optimal_path_nodes = set([node.City for node in optimal_path])

    # Get nodes connected to the source and destination cities
    connected_nodes = set()
    for edge in G.edges():
        if edge[0] in optimal_path_nodes:
            connected_nodes.add(edge[1])
        if edge[1] in optimal_path_nodes:
            connected_nodes.add(edge[0])

    # Include source and destination cities and a few additional cities
    additional_cities = set()
    for node in optimal_path:
        additional_cities.update(G.neighbors(node.City))
    additional_cities = additional_cities.difference(optimal_path_nodes)

    # Select a subset of 8 cities
    displayed_cities = optimal_path_nodes.union(connected_nodes).union(additional_cities)
    displayed_cities = list(displayed_cities)[:10]

    # Add fake routes
    for route in fake_routes:
        for i in range(len(route)-1):
            G.add_edge(route[i], route[i+1], weight=np.random.randint(20, 100))

    # Subgraph containing only selected cities
    H = G.subgraph(displayed_cities)

    # Draw subgraph
    pos = nx.spring_layout(H)
    nx.draw(H, pos, with_labels=True, node_color='skyblue', node_size=2000, arrowsize=20)

    # Highlight optimal path
    optimal_path_edges = [(optimal_path[i].City, optimal_path[i+1].City) for i in range(len(optimal_path)-1)]
    nx.draw_networkx_edges(H, pos, edgelist=optimal_path_edges, edge_color='red', width=2)

    # Add labels for cities
    for city in H.nodes():
        x, y = pos[city]
        plt.text(x, y, city, fontsize=12, ha='center', va='center', color='black')

    plt.title(f"Optimal Path with Cost: {min_cost}", fontsize=14)
    plt.axis('off')  # Turn off axis
    plt.show()

def on_click():
    source = source_var.get()
    destination = destination_var.get()
    package_priority = priority_var.get()

    start_node = Node(source, None, 0, 0, package_priority)
    goal_node = Node(destination, None, 0, 0, package_priority)

    all_paths = best_first_search(start_node, goal_node, heuristic, df)

    if all_paths:
        min_cost = float('inf')
        optimal_path = None
        for path in all_paths:
            total_cost = 0
            for idx, node in enumerate(path[:-1]):
                next_node = path[idx + 1]
                cost = get_cost(node, next_node)
                total_cost += cost
            if total_cost < min_cost:
                min_cost = total_cost
                optimal_path = path

        optimal_path_str = ' -> '.join([node.City for node in optimal_path] + [goal_node.City])
        messagebox.showinfo("Optimal Path", f"Optimal Path: {optimal_path_str}\nOptimal Cost: {min_cost}")

        # Generate fake routes
        fake_routes = generate_fake_routes(source, destination)

        # Plot graph with fake routes
        plot_graph(df, optimal_path, min_cost, fake_routes)
    else:
        messagebox.showinfo("No Routes Found", "No routes found.")

# Read data from CSV
df = pd.read_csv('dl.csv')

# Create GUI
root = tk.Tk()
root.title("Best-First Search GUI")

source_label = ttk.Label(root, text="Source City:")
source_label.grid(row=0, column=0)
source_var = tk.StringVar()
source_entry = ttk.Entry(root, textvariable=source_var)
source_entry.grid(row=0, column=1)

destination_label = ttk.Label(root, text="Destination City:")
destination_label.grid(row=1, column=0)
destination_var = tk.StringVar()
destination_entry = ttk.Entry(root, textvariable=destination_var)
destination_entry.grid(row=1, column=1)

priority_label = ttk.Label(root, text="Package Priority (1-10):")
priority_label.grid(row=2, column=0)
priority_var = tk.IntVar()
priority_spinbox = ttk.Spinbox(root, from_=1, to=10, textvariable=priority_var)
priority_spinbox.grid(row=2, column=1)

search_button = ttk.Button(root, text="Search", command=on_click)
search_button.grid(row=3, columnspan=2)

root.mainloop()
