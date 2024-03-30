from datetime import datetime as dt, timedelta
from process_data import coordinates, get_graph, get_average_speed
import heapq
import math
from typing import List, Callable, Tuple
import time

def calculate_time_difference(arrival_time_1, arrival_time_2):
    return (arrival_time_2 - arrival_time_1).total_seconds()

def dijkstra(graph, start, end, start_time):
    
    pq = []  
    heapq.heappush(pq, (0, (start, start_time, "")))  
    visited = set() 
    distances = {node: float('inf') for node in graph}  
    distances[start] = 0
    previous = {}  
    
    while pq:
        current_distance, (current_node, current_time, current_line) = heapq.heappop(pq)
        if current_node == end:
            path = []
            while (current_node, current_time) in previous:
                path.insert(0, (current_node, current_time.strftime("%H:%M:%S"), current_line))
                current_node, current_time, current_line = previous[current_node, current_time]
            path.insert(0, (start, current_time.strftime("%H:%M:%S"), ""))
            return path, distances[end]

        if (current_node, current_time) in visited:
            continue

        visited.add((current_node, current_time))

        for next_node, line_departures in graph[current_node].items():
            
            for line, departure, arrival in line_departures:
                
                if departure > current_time:  # Ensure we don't select past departures
                    distance = current_distance + calculate_time_difference(current_time, arrival)
                    
                    if distance < distances[next_node]:
                        distances[next_node] = distance
                        previous[(next_node, arrival)] = (current_node, current_time, line)
                        heapq.heappush(pq, (distance, (next_node, arrival, line)))

    return None, float('inf')  


# stores the information about each stop, assuming a bus is leaving from some stop (information contained in the graph)
# the node denotes what stop one departure arrives at 
# so a person departs from the previous stop (departure time) and arrives at Node (arrival time) using a bus/tram of line 
class Node:
    def __init__(self, name: str, line: str, departure :dt, arrival: dt, g=0, h=0, f=0) -> None:
          self.name = name
          self.line = line
          self.departure = departure
          self.arrival = arrival
          self.g = g
          self.h = h
          self.f = f
    def __str__(self) -> str:
         hour = self.departure.strftime("%H:%M:%S")
         arr = self.arrival.strftime("%H:%M:%S")
         return f"To: {self.name}  === at { hour} with line {self.line}, will arrive at {arr}"


def euclidian_heuristic(current_node : Node, goal: str) -> float:
     start_long = coordinates[current_node.name][0]
     start_lat = coordinates[current_node.name][1]

     end_long = coordinates[goal][0]
     end_lat = coordinates[goal][1]

     return math.sqrt((end_long - start_long)**2 + (end_lat - start_lat)**2)


def euclidian_heuristic2(current_node : str, goal: str) -> float:
     start_long = coordinates[current_node][0]
     start_lat = coordinates[current_node][1]

     end_long = coordinates[goal][0]
     end_lat = coordinates[goal][1]

     return math.sqrt((end_long - start_long)**2 + (end_lat - start_lat)**2)

def manhattan_heuristic(current_node : Node, goal: str) -> float:
     start_long = coordinates[current_node.name][0]
     start_lat = coordinates[current_node.name][1]

     end_long = coordinates[goal][0]
     end_lat = coordinates[goal][1]

     return abs(start_long - end_long)+abs(start_lat - end_lat)


def manhattan_heuristic2(current_node : str, goal: str) -> float:
     start_long = coordinates[current_node][0]
     start_lat = coordinates[current_node][1]

     end_long = coordinates[goal][0]
     end_lat = coordinates[goal][1]

     return abs(start_long - end_long)+abs(start_lat - end_lat)



     


# g function
def cost(prev: Node, current: Node, calculation_type="t"):
    if calculation_type=="t":
         return (current.arrival - prev.arrival).total_seconds()
    else:
         return 0 if prev.line==current.line or prev.line==0 else 1


# finding th neighborhood of one node - list of all possible routes one can take from the current bus stop node.name
def neighborhood(node: Node) -> List[Node]:
    lst = []
    for stop, schedule in graph[node.name].items():
          
          for line, departure, arrival in schedule:
               
               # consider only the lines we can take 
               if(node.arrival < departure) and (node.arrival - departure < timedelta(hours=1)):
                    lst.append(Node(stop, line, departure=departure, arrival=arrival))
    return lst
     

def a_star2( graph, start, end, start_time, heuristic:Callable[[str, str], float], coefficient:float = 1.0, parameter:str="t") -> Tuple[List[Node], int]:
    pq = []  
    heapq.heappush(pq, (0, (start, start_time, "")))  
    visited = set() 
    distances = {node: float('inf') for node in graph}  
    distances[start] = 0
    previous = {}  
    
    while pq:
        current_distance, (current_node, current_time, current_line) = heapq.heappop(pq)
        if current_node == end:
            path = []
            while (current_node, current_time) in previous:
                path.insert(0, (current_node, current_time.strftime("%H:%M:%S"), current_line))
                current_node, current_time, current_line = previous[current_node, current_time]
            path.insert(0, (start, current_time.strftime("%H:%M:%S"), ""))
            return path, distances[end]

        if (current_node, current_time) in visited:
            continue

        visited.add((current_node, current_time))

        for next_node, line_departures in graph[current_node].items():
            
            for line, departure, arrival in line_departures:
                
                if departure > current_time:  # Ensure we don't select past departures
                    distance = current_distance + calculate_time_difference(current_time, arrival) + coefficient*heuristic(current_node, next_node)
                    
                    if distance < distances[next_node]:
                        distances[next_node] = distance
                        previous[(next_node, arrival)] = (current_node, current_time, line)
                        heapq.heappush(pq, (distance, (next_node, arrival, line)))

    return None, float('inf')  


def a_star( start, end, start_time, heuristic:Callable[[Node, str], float], coefficient:float = 1.0, parameter:str="t") -> Tuple[List[Node], int]:
    open_set = [Node(start, line="", departure=start_time, arrival=start_time)]
    #  heapq.heappush(open_set, Node(start, start="", departure=start_time, arrival=start_time))
    closed_set = []

    while open_set:
        node = None
        node_cost = float('inf')

        for testing_node in open_set:
            if testing_node.f <  node_cost:
                    node = testing_node
                    node_cost = testing_node.f
        if node.name == end:
             closed_set.append(node)
             break
        
        open_set.remove(node)
        closed_set.append(node)

        for next_node in neighborhood(node):
            if next_node not in open_set and next_node not in closed_set:
                  open_set.append(next_node)
                  next_node.h = coefficient*heuristic(next_node, end)
                  # time difference between arrivals instead of departure and arrival at node 
                  # to count the time waited at a stop
                  next_node.g = node.g + cost(node, next_node, calculation_type=parameter)
                  next_node.f = next_node.h + next_node.g
            else:
                 if next_node.g > node.g + cost(node, next_node, calculation_type=parameter):
                      next_node.g = node.g + cost(node, next_node, calculation_type=parameter)
                      next_node.f = coefficient*next_node.h + next_node.g

                      if next_node in closed_set:
                           open_set.append(next_node)
                           closed_set.remove(next_node)

    return closed_set, closed_set[-1].g


def get_a_star_path(graph, start_stop: str, end_stop: str, start_time: dt, coeff: float,h_func: Callable[[Node, str], float]=euclidian_heuristic, parameter:str="t"):
    
    begin_computation = time.time()
    path, a_star_time = a_star2(graph, start_stop, end_stop, start_time, heuristic=h_func, coefficient=coeff, parameter=parameter)
    end_computation = time.time()
    execution_time = round(end_computation-begin_computation, 5)
    
    print(f"\n======== A* Execution time:{execution_time}\t coefficient={coeff}\t h={ h_func.__name__}\n")


    if path:
        for i in range(1, len(path)):
                name_prev = path[i-1][0]
                
                name,  arrives, line = path[i]
                print(f"{name_prev}  with  {line}   -->  {name} at {arrives} ")
        print("Total time: ", a_star_time )
    else:
            print("No path found from", start_stop, "to", end_stop)

def get_dijkstra(graph, start, end, start_time):
    
    begin_computation = time.time()
    path, total_time = dijkstra(graph, start, end, start_time)
    end_computation = time.time()
    execution_time = round(end_computation-begin_computation, 5)

    if path:
        print(f"\n======== Dijkstra algorithm execution time: {execution_time}\n")
        for i in range(1, len(path)):
                name_prev = path[i-1][0]
                
                name,  arrives, line = path[i]
                print(f"{name_prev}  with  {line}   -->  {name} at {arrives} ")
        print("Total time: ", total_time )
    else:
            print("No path found from", start_stop, "to", end_stop)


if __name__=="__main__":

    start_stop = 'Katedra'  # Starting stop
    end_stop = 'Kliniki - Politechnika Wrocławska'    # Ending stop
    start_time = dt.strptime('1900-01-01 09:20:00', '%Y-%m-%d %H:%M:%S')  # Starting time
    graph = get_graph()
    

    # get_a_star_path(start_stop, end_stop, start_time,coeff=1)
    # get_a_star_path(start_stop, end_stop, start_time,coeff=1/get_average_speed())
    get_dijkstra(graph, start_stop, end_stop, start_time)
    get_a_star_path(graph, start_stop, end_stop, start_time, coeff=10, h_func=euclidian_heuristic2)
    get_a_star_path(graph, start_stop, end_stop, start_time, coeff=1000, h_func=euclidian_heuristic2)
    get_a_star_path(graph, start_stop, end_stop, start_time, coeff=10000, h_func=euclidian_heuristic2)
    get_a_star_path(graph, start_stop, end_stop, start_time, coeff=1000000, h_func=euclidian_heuristic2)
    get_a_star_path(graph, start_stop, end_stop, start_time, coeff=1/get_average_speed(), h_func=euclidian_heuristic2)



    print("\n\nManhattan heuristic")
    get_a_star_path(graph, start_stop, end_stop, start_time, coeff=10, h_func=manhattan_heuristic2)
    get_a_star_path(graph, start_stop, end_stop, start_time, coeff=1000, h_func=manhattan_heuristic2)
    get_a_star_path(graph, start_stop, end_stop, start_time, coeff=10000, h_func=manhattan_heuristic2)
    get_a_star_path(graph, start_stop, end_stop, start_time, coeff=1000000, h_func=manhattan_heuristic2)
    
    


    # get_a_star_path(start_stop, end_stop, start_time,coeff=1, parameter="p")
    # the given path will be in a tuple (name of stop, )
    # get_dijkstra(graph, "Katedra", "Kliniki - Politechnika Wrocławska", start_time)

    # get_a_star_path("Katedra", "Kliniki - Politechnika Wrocławska", start_time, 1)
    # get_a_star_path("Katedra", "Kliniki - Politechnika Wrocławska", start_time, 10)
    # get_a_star_path("Katedra", "Kliniki - Politechnika Wrocławska", start_time, 100)
    # get_a_star_path("Katedra", "Kliniki - Politechnika Wrocławska", start_time, 1000)
    # get_a_star_path("Katedra", "Kliniki - Politechnika Wrocławska", start_time, 10000)
    # get_a_star_path("Katedra", "Kliniki - Politechnika Wrocławska", start_time, 100000)
    # get_a_star_path("Katedra", "Kliniki - Politechnika Wrocławska", start_time, 1000000)
    # get_a_star_path("Katedra", "Kliniki - Politechnika Wrocławska", start_time, 1000000, manhattan_heuristic)

    # get_a_star_path("DWORZEC NADODRZE", "Niedźwiedzia", start_time, 1000000)
    # get_a_star_path("DWORZEC NADODRZE", "Niedźwiedzia", start_time, 1000000, manhattan_heuristic)

    # get_a_star_path("Chełmońskiego", "Dubois", start_time, 1000000)
    # get_a_star_path("Chełmońskiego", "Dubois", start_time, 1000000, manhattan_heuristic)
    #get_a_star_path("Katedra", "Kliniki - Politechnika Wrocławska", start_time, get_average_speed())
    #get_a_star_path("Katedra", "Kliniki - Politechnika Wrocławska", start_time, 10000)


    # print("Avg speed;  (takes a long time to compute A*): ",get_average_speed())
    

    # # get_a_star_path("PL. GRUNWALDZKI", "Niedźwiedzia", start_time, 10000000)
    # get_dijkstra(graph, "PL. GRUNWALDZKI", "Niedźwiedzia", start_time)

    