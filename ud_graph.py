# Course: CS 261 - Data Structures
# Author: Matthew DeMichele
# Assignment: Assignment 6
# Description: Implementation of an undirected graph

class Stack:
    """
    Class implementing STACK ADT.
    Supported methods are: push, pop, top, is_empty

    DO NOT CHANGE THIS CLASS IN ANY WAY
    YOU ARE ALLOWED TO CREATE AND USE OBJECTS OF THIS CLASS IN YOUR SOLUTION
    """
    def __init__(self):
        """ Initialize empty stack based on Python list """
        self._data = []

    def push(self, value: object) -> None:
        """ Add new element on top of the stack """
        self._data.append(value)

    def pop(self):
        """ Remove element from top of the stack and return its value """
        return self._data.pop()

    def top(self):
        """ Return value of top element without removing from stack """
        return self._data[-1]

    def is_empty(self):
        """ Return True if the stack is empty, return False otherwise """
        return len(self._data) == 0

    def __str__(self):
        """ Return content of the stack as a string (for use with print) """
        data_str = [str(i) for i in self._data]
        return "STACK: { " + ", ".join(data_str) + " }"
        
class Queue:
    """
    Class implementing QUEUE ADT.
    Supported methods are: enqueue, dequeue, is_empty

    DO NOT CHANGE THIS CLASS IN ANY WAY
    YOU ARE ALLOWED TO CREATE AND USE OBJECTS OF THIS CLASS IN YOUR SOLUTION
    """
    def __init__(self):
        """ Initialize empty queue based on Python list """
        self._data = []

    def enqueue(self, value: object) -> None:
        """ Add new element to the end of the queue """
        self._data.append(value)

    def dequeue(self):
        """ Remove element from the beginning of the queue and return its value """
        return self._data.pop(0)

    def is_empty(self):
        """ Return True if the queue is empty, return False otherwise """
        return len(self._data) == 0

    def __str__(self):
        """ Return content of the stack as a string (for use with print) """
        data_str = [str(i) for i in self._data]
        return "QUEUE { " + ", ".join(data_str) + " }"    

class UndirectedGraph:
    """
    Class to implement undirected graph
    - duplicate edges not allowed
    - loops not allowed
    - no edge weights
    - vertex names are strings
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency list
        """
        self.adj_list = dict()

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            for u, v in start_edges:
                self.add_edge(u, v)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        """
        out = [f'{v}: {self.adj_list[v]}' for v in self.adj_list]
        out = '\n  '.join(out)
        if len(out) < 70:
            out = out.replace('\n  ', ', ')
            return f'GRAPH: {{{out}}}'
        return f'GRAPH: {{\n  {out}}}'

    # ------------------------------------------------------------------ #

    def add_vertex(self, v: str) -> None:
        """
        Add new vertex to the graph
        """
        # If the vertex does not exist already, add it to the adjacency list 
        if self.adj_list.get(v) == None:
            self.adj_list[v] = []
            return None 
        # If the vertex already exists, do nothing 
        else:
            return None
        
    def add_edge(self, u: str, v: str) -> None:
        """
        Add edge to the graph
        """
        # CASE 01: If u and v are the same, do nothing 
        if u == v:
            return None 
            
        # CASE 02: If both vertices don't exist
        elif self.adj_list.get(u) == None and self.adj_list.get(v) == None:
            self.adj_list[u] = [v]
            self.adj_list[v] = [u]
        
        # CASE 03: If vertice u doesn't exist
        elif self.adj_list.get(u) == None:
            self.adj_list[u] = [v]
            self.adj_list[v].append(u)
        
        # CASE 04: If vertice v doesn't exist 
        elif self.adj_list.get(v) == None:
            self.adj_list[v] = [u]
            self.adj_list[u].append(v)
            
        else:
            # Search all the vertices in vertice u stack
            for vertice in range(0, len(self.adj_list[u])):
                if self.adj_list[u][vertice] == v:
                    return None 
            # If not found, add vertice to both stacks 
            self.adj_list[u].append(v)
            self.adj_list[v].append(u)

    def remove_edge(self, v: str, u: str) -> None:
        """
        Remove edge from the graph
        """
        # EDGE CASE 1: if one or both of the vertices are missing, do nothing 
        if self.adj_list.get(v) == None or self.adj_list.get(u) == None:
            return None 
        
        # EDGE CASE 2: if one or both of the vertices don't have any edges yet, do nothing 
        elif len(self.adj_list[u]) == 0 or len(self.adj_list[v]) == 0:
            return None
            
        else:
            # Search u for v, if found, remove 
            deleted = 0
            index = 0
            while index < len(self.adj_list[u]) and deleted == 0:
                if self.adj_list[u][index] == v:
                    del self.adj_list[u][index]
                    deleted = 1
                index += 1
                
            # Search v for u, if found, remove 
            deleted = 0
            index = 0
            while index < len(self.adj_list[v]) and deleted == 0:
                if self.adj_list[v][index] == u:
                    del self.adj_list[v][index]
                    deleted = 1
                index += 1
                    
            return None

    def remove_vertex(self, v: str) -> None:
        """
        Remove vertex and all connected edges
        """
        # EDGE CASE: If graph doesn't have vertex, do nothing 
        if self.adj_list.get(v) == None:
            return None 
        else:
            # Remove v from all other the nodes who have v in their adjacency lists 
            connected = []
            for index in range(0, len(self.adj_list[v])):
                connected.append(self.adj_list[v][index])
                
            if len(connected) > 0:
                for index in range(0, len(connected)):
                
                    current_vertex = self.adj_list[connected[index]]
                    current_vertex.remove(v)
            
            # Remove vertex itself from adj_list
            self.adj_list.pop(v)
            return None        

    def get_vertices(self) -> []:
        """
        Return list of vertices in the graph (any order)
        """
        # Create an empty list to be returned 
        vertices = []
        
        # EDGE CASE 1: No vertices 
        if len(self.adj_list) == 0:
            return vertices 
        
        # Loop through each vertex in the adjacency list and add it to the vertices list
        for key in self.adj_list:
            vertices.append(key)
                
        return vertices

    def get_edges(self) -> []:
        """
        Return list of edges in the graph (any order)
        """
        # Create an empty list to be returned 
        edges = []
        
        # Loop through each vertex in the adjacency list 
        for key in self.adj_list:
            
            current_vertices = self.adj_list[key]
            
            for edge in range(0, len(current_vertices)):
                
                current_edge = (key, current_vertices[edge])
                
                index = 0
                duplicate = 0
                
                while index < len(edges) and duplicate == 0:
                    
                    # CASE 01: If tuples are exactly the same
                    if current_edge[0] == edges[index][0] and current_edge[1] == edges[index][1]:
                        duplicate = 1
                    # CASE 02: If tuples are same but in reverse order 
                    elif current_edge[0] == edges[index][1] and current_edge[1] == edges[index][0]:
                        duplicate = 1
                    # CASE 03: Tuples aren't matching, continue searching for duplicates
                    else:
                        index += 1
                
                if duplicate == 0:
                    edges.append(current_edge)
                    
        return edges

    def is_valid_path(self, path: []) -> bool:
        """
        Return true if provided path is valid, False otherwise
        """
        
        # EDGE CASE 1: Empty path 
        if len(path) == 0:
            return True 
            
        # EDGE CASE 2: Single path on single graph node 
        if len(path) == 1:
            if self.adj_list.get(path[0]) != None:
                return True
            else:
                return False
        
        # To check for valid path:
        # 01. Start with first letter as first vertex key. 
        # 02. Second letter starts as first vertex value. 
        # 03. If value is found, value becomes new key 
        # 04. Search for next value in next key
        # 05. Continue until end of string 
        index1 = 0
        invalid = 1
        
        while index1 < len(path) - 1 and invalid == 1:
            
            key = path[index1]
            
            found = 0
            index2 = 0
            
            # EDGE CASE 3: If a vertex in the path has no edges, it's not a valid path 
            if len(self.adj_list[key]) == 0:
                return False 
                
            # Iterate through key in adj_list, searching for the next vertex in path 
            while index2 < len(self.adj_list[key]) and found == 0:
                
                if self.adj_list[key][index2] == path[index1 + 1]:
                    found = 1
                    
                index2 += 1
                
            # If we didn't find the next vertex in our key list, it's not a valid path 
            if found == 0:
                return False 
            else:
                index1 += 1
                
        # If we make it through the entire path, it's valid
        return True
               
    def dfs(self, v_start, v_end=None) -> []:
        """
        Return list of vertices visited during DFS search
        Vertices are picked in alphabetical order
        """
        # EDGE CASE 1: v_start is not in graph 
        if v_start not in self.adj_list:
            return []
        
        # Empty set of visited vertices 
        visited = set()
        
        # empty stack 
        stack = Stack()
        
        # Final list to return 
        return_list = []
        
        # add v_start to stack 
        stack.push(v_start)
        
        while stack.is_empty() == False:
            
            # Pop the vertex at the top of the stack
            vertex = stack.pop()
            
            # If vertex hasn't been visited already
            if vertex not in visited:
                visited.add(vertex)
                
                # Add that vertex to the return list 
                return_list.append(vertex)
                
                # Check if you've hit the end vertex yet:
                if vertex == v_end:
                    return return_list
                
                # Push all the direct successors in a list 
                successors = []
                for index in range(0, len(self.adj_list[vertex])):
                    successors.append(self.adj_list[vertex][index])
                    
                # Sort list alphabetically 
                for index in range(0, len(successors)):
                    current = successors[index]
                
                    while index > 0:
                        # Swap if value is greater than previous 
                        if successors[index] > successors[index - 1]:
                            successors[index], successors[index - 1] = successors[index - 1], successors[index]
                        else:
                            break 
                        index -= 1
                
                # Push all items in successor list into the stack 
                for index in range(0, len(successors)):
                    stack.push(successors[index])
        
        return return_list
       
    def bfs(self, v_start, v_end=None) -> []:
        """
        Return list of vertices visited during BFS search
        Vertices are picked in alphabetical order
        """
        # EDGE CASE 1: v_start is not in graph 
        if v_start not in self.adj_list: 
            return []
            
        # Initialize visited set 
        visited = set()
            
        # Initialize an empty queue 
        queue = Queue()
        
        # Initialize our list to return 
        return_list = []
        
        # Add v_start to queue 
        queue.enqueue(v_start)
        
        while queue.is_empty() == False: 
            
            # Dequeue the first vertex in queue 
            vertex = queue.dequeue()
            
            # If vertex hasn't been visited yet, perform operations
            if vertex not in visited:
                # Add vertex to visited set
                visited.add(vertex)
                # Add vertex to the return list 
                return_list.append(vertex)
                
                # If you've hit the v_end, end traversal:
                if vertex == v_end:
                    return return_list 
                    
                # Push direct successor into queue in alphabetical order 
                successors = []
                for index in range(0, len(self.adj_list[vertex])):
                    successors.append(self.adj_list[vertex][index])
                    
                # Sort in alphabetical order (insertion sort)
                for index in range(0, len(successors)):
                    
                    while index > 0:
                        if successors[index] < successors[index - 1]:
                            successors[index], successors[index -1] = successors[index - 1], successors[index]
                        else:
                            break 
                        index -= 1
                
                # enqueue into queue 
                for index in range(0, len(successors)):
                    queue.enqueue(successors[index])
                    
        return return_list

    def node_array(self, nodes) -> []:
        """Helper function for count_connected_components. Simply takes a dictionary of keys and returns and array of keys"""
        node_array = []
        
        for node in nodes:
            node_array.append(node)
        
        return node_array

    def count_connected_components(self):
        """
        Return number of connected componets in the graph
        """
        # EDGE CASE 1: Graph has no vertices 
        if len(self.adj_list) == 0:
            return 0
            
        # Initialize compnonent number 
        components = 0
        
        # Initialize visited set 
        visited = set()
        
        # Start with first vertex and traverse all of its connected nodes 
        nodes = self.adj_list.keys()
        
        node_array = self.node_array(nodes)
        
        while len(visited) != len(node_array):
            
            # Find correct node to start traversal
            index = 0
            while node_array[index] in visited:
                index += 1
                
            # Get an array of connected nodes  
            connected = self.dfs(node_array[index])
            
            # push our connected nodes into visited 
            for index in range(0, len(connected)):
                visited.add(connected[index])
                
            # Add 1 to our components count 
            components += 1
            
        return components
            
    def has_cycle(self):
        """
        Return True if graph contains a cycle, False otherwise
        """
        # Initialize empty set for the visited vertices
        visited = set()
       
        # Initialize empty stack to process vertices 
        stack = Stack()
       
        # Get the first vertex
        node_dict = self.adj_list.keys()
        node_list = self.node_array(node_dict)
       
        # push first vertex into stack 
        for vertex in range(0, len(self.adj_list)):
            stack.push(node_list[vertex])
        
            # neighbors
            neighbors = set()
       
            # Continue executing until you've hit every vertex
            while stack.is_empty() == False:
                vertex = stack.pop()
           
                # print("vertex: " + str(vertex))
                if vertex not in visited:
                    visited.add(vertex)
               
                    for index in range(0, len(self.adj_list[vertex])):
                        curr = self.adj_list[vertex][index]
                    
                        if curr in neighbors and curr not in visited:
                            return True 
                        else:
                            neighbors.add(curr)
                            stack.push(curr)
                   
        return False
               
if __name__ == '__main__':

    print("\nPDF - method add_vertex() / add_edge example 1")
    print("----------------------------------------------")
    g = UndirectedGraph()
    print(g)
    
    for v in 'ABCDE':
        g.add_vertex(v)
    print(g)
    
    g.add_vertex('A')
    print(g)
    
    for u, v in ['DD','AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE', ('B', 'C')]:
        g.add_edge(u, v)
    print(g)
    
    
    print("\nPDF - method remove_edge() / remove_vertex example 1")
    print("----------------------------------------------------")
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    g.remove_vertex('DOES NOT EXIST')
    g.remove_edge('A', 'B')
    g.remove_edge('X', 'B')
    print(g)
    g.remove_vertex('D')
    print(g)
    
    
    print("\nPDF - method get_vertices() / get_edges() example 1")
    print("---------------------------------------------------")
    g = UndirectedGraph()
    print(g.get_edges(), g.get_vertices(), sep='\n')
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE'])
    print(g.get_edges(), g.get_vertices(), sep='\n')
    
    
    print("\nPDF - method is_valid_path() example 1")
    print("--------------------------------------")
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    test_cases = ['ABC', 'ADE', 'ECABDCBE', 'ACDECB', '', 'D', 'Z']
    for path in test_cases:
        print(list(path), g.is_valid_path(list(path)))
    
    
    print("\nPDF - method dfs() and bfs() example 1")
    print("--------------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = 'ABCDEGH'
    for case in test_cases:
        print(f'{case} DFS:{g.dfs(case)} BFS:{g.bfs(case)}')
    print('-----')
    for i in range(1, len(test_cases)):
        v1, v2 = test_cases[i], test_cases[-1 - i]
        print(f'{v1}-{v2} DFS:{g.dfs(v1, v2)} BFS:{g.bfs(v1, v2)}')
    
    
    print("\nPDF - method count_connected_components() example 1")
    print("---------------------------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print(g.count_connected_components(), end=' ')
    print()
    
    
    print("\nPDF - method has_cycle() example 1")
    print("----------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG',
        'add FG', 'remove GE')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print('{:<10}'.format(case), g.has_cycle())
