# Course: CS261 - Data Structures
# Author: Matthew DeMichele
# Assignment: Assignment 6
# Description: Implementation of a directed graph

class DynamicArrayException(Exception):
    pass

class DynamicArray:
    """
    Class implementing a Dynamic Array
    Supported methods are:
    append, pop, swap, get_at_index, set_at_index, length

    DO NOT CHANGE THIS CLASS IN ANY WAY
    YOU ARE ALLOWED TO CREATE AND USE OBJECTS OF THIS CLASS IN YOUR SOLUTION
    """

    def __init__(self, arr=None):
        """ Initialize new dynamic array """
        self.data = arr.copy() if arr else []

    def __iter__(self):
        """
        Disable iterator capability for DynamicArray class
        This means loops and aggregate functions like
        those shown below won't work:

        arr = StaticArray()
        for value in arr:     # will not work
        min(arr)              # will not work
        max(arr)              # will not work
        sort(arr)             # will not work
        """
        return None

    def __str__(self) -> str:
        """ Return content of dynamic array in human-readable form """
        return str(self.data)

    def append(self, value: object) -> None:
        """ Add new element at the end of the array """
        self.data.append(value)

    def pop(self) -> object:
        """ Removes element from end of the array and return it """
        return self.data.pop()

    def swap(self, i: int, j: int) -> None:
        """ Swaps values of two elements given their indicies """
        self.data[i], self.data[j] = self.data[j], self.data[i]

    def get_at_index(self, index: int) -> object:
        """ Return value of element at a given index """
        if index < 0 or index >= self.length():
            raise DynamicArrayException
        return self.data[index]

    def __getitem__(self, index: int) -> object:
        """ Return value of element at a given index using [] syntax """
        return self.get_at_index(index)

    def set_at_index(self, index: int, value: object) -> None:
        """ Set value of element at a given index """
        if index < 0 or index >= self.length():
            raise DynamicArrayException
        self.data[index] = value

    def __setitem__(self, index: int, value: object) -> None:
        """ Set value of element at a given index using [] syntax """
        self.set_at_index(index, value)

    def length(self) -> int:
        """ Return the length of the DA """
        return len(self.data)

class MinHeap:
    def __init__(self, start_heap=None):
        """
        Initializes a new MinHeap
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.heap = DynamicArray()

        # populate MH with initial values (if provided)
        # before using this feature, implement add() method
        if start_heap:
            for node in start_heap:
                self.add(node)

    def __str__(self) -> str:
        """
        Return MH content in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        return 'HEAP ' + str(self.heap)

    def is_empty(self) -> bool:
        """
        Return True if no elements in the heap, False otherwise
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        return self.heap.length() == 0

    def add(self, node: object) -> None:
        """
        Method adds a new item to the heap maintaining the heap property
        """
        # EDGE CASE 1: No items in heap 
        if self.heap.length() == 0:
            self.heap.append(node)
            return None
            
        # 01. Insert node at end of array 
        self.heap.append(node)
        # 02. Find inserted node's parent node
        index = self.heap.length() - 1
        
        while index > 0:
            parent = int((index - 1) / 2)
        # 03. Compare node and parent node, if less than parent, swap 
            if self.heap.get_at_index(parent)[1] > self.heap.get_at_index(index)[1]:
                temp = self.heap.get_at_index(parent)
                self.heap.set_at_index(parent, self.heap.get_at_index(index))
                self.heap.set_at_index(index, temp)
                index = parent
            else:
                index = 0
        return None

    def get_min(self) -> object:
        """
        Method gets the item with the minimum key without removing it from the heap 
        """
        # EXCEPTION 1: Heap is empty 
        if self.heap.length() == 0:
            raise MinHeapException("Heap is Empty")
        
        # Simply return the first element in DynamicArray
        return self.heap.get_at_index(0)

    def child_min(self, value1, value2):
        """
        Helper function for remove_min. Just returns the smaller of two values 
        """
        if value1[1] > value2[1]:
            return value2
        else:
            return value1
            
    def child_index(self, value1, value2, index1, index2):
        """
        Helper function for remove_min. Just returns the index of the smaller of two values 
        """
        if value1[1] > value2[1]:
            return index2
        else:
            return index1
    
    def remove_min(self) -> object:
        """
        Method returns item with the minimum key and removes it from the heap
        """
        # EXCEPTION 1: If heap is empty, raise exception 
        if self.heap.length() == 0:
            raise MinHeapException("heap is empty")
        # Find min value and remember it 
        min = self.heap.get_at_index(0)
        
        # Swap first and last values 
        self.heap.set_at_index(0, self.heap.get_at_index(self.heap.length() - 1))
        
        # Remove last element 
        self.heap.pop()
        
        # Start with first element and its children 
        current = 0
        left_child = (2 * current) + 1
        right_child = (2 * current) + 2
        
        while left_child < self.heap.length() or right_child < self.heap.length():
            
            # CASE 1: There is a left child and a right child 
            if left_child < self.heap.length() and right_child < self.heap.length() and (self.heap.get_at_index(current)[1] > self.heap.get_at_index(left_child)[1] or self.heap.get_at_index(current)[1] > self.heap.get_at_index(right_child)[1]):
                # Find the minimum of the two children and its index 
                min_child = self.child_min(self.heap.get_at_index(left_child), self.heap.get_at_index(right_child))
                min_index = self.child_index(self.heap.get_at_index(left_child), self.heap.get_at_index(right_child), left_child, right_child)
                
                if self.heap.get_at_index(current)[1] > min_child[1]:
                    # swap the two elements 
                    temp = min_child 
                    self.heap.set_at_index(min_index, self.heap.get_at_index(current))
                    self.heap.set_at_index(current, temp)
                
                    # Repeat process with new index and its children 
                    current = min_index
                    left_child = (2 * current) + 1
                    right_child = (2 * current) + 2
            
            # CASE 2: There is only a left child and swap is still needed
            elif left_child < self.heap.length() and self.heap.get_at_index(current)[1] > self.heap.get_at_index(left_child)[1]:
                # Swap 
                temp = self.heap.get_at_index(left_child)
                self.heap.set_at_index(left_child, self.heap.get_at_index(current))
                self.heap.set_at_index(current, temp)
                
                current = left_child 
                left_child = (2 * current) + 1
                right_child = (2 * current) + 2
                    
            # CASE 3: There is only a right child and swap is still needed
            elif right_child < self.heap.length() and self.heap.get_at_index(current)[1] > self.heap.get_at_index(right_child)[1]:
                # Swap 
                temp = self.heap.get_at_index(right_child)
                self.heap.set_at_index(right_child, self.heap.get_at_index(current))
                self.heap.set_at_index(current, temp)
                
                current = right_child 
                left_child = (2 * current) + 1
                right_child = (2 * current) + 2
            
            # CASE 4: Current value is in the right spot 
            else:
                return min 
        
        return min
            

    def build_heap(self, da: DynamicArray) -> None:
        """
        Method builds a min heap from a dynamicArray
        """
        # Start with the first non-leaf element in the heap 
        current = int(da.length() / 2) - 1
        
        # Percolate up the tree until you hit the root element 
        while current >= 0:
            temp = current 
            
            left_child = (2 * current) + 1
            right_child = (2 * current) + 2
            
            while left_child < da.length() or right_child < da.length():
                
                # CASE 1: There is a left AND right child AND swap is needed 
                if left_child < da.length() and right_child < da.length() and (da.get_at_index(current) > da.get_at_index(left_child) or da.get_at_index(current) > da.get_at_index(right_child)):
                    min_child = self.child_min(da.get_at_index(left_child), da.get_at_index(right_child))
                    min_index = self.child_index(da.get_at_index(left_child), da.get_at_index(right_child), left_child, right_child)
                    
                    # Swap if current element is greater than minimum of the two children
                    if da.get_at_index(current) > min_child:
                        temp2 = da.get_at_index(current)
                        da.set_at_index(current, min_child)
                        da.set_at_index(min_index, temp2)
                        
                        current = min_index
                        left_child = (2 * current) + 1
                        right_child = (2 * current) + 2
                        
                # CASE 2: There is a left child AND swap is needed 
                elif left_child < da.length() and da.get_at_index(current) > da.get_at_index(left_child):
                    # Swap current element and left child 
                    temp2 = da.get_at_index(current)
                    da.set_at_index(current, da.get_at_index(left_child))
                    da.set_at_index(left_child, temp2)
                    
                    current = left_child 
                    left_child = (2 * current) + 1 
                    right_child = (2 * current) + 2
                    
                # CASE 3: There is a right child AND swap is needed 
                elif right_child < da.length() and da.get_at_index(current) > da.get_at_index(right_child):
                    # Swap current element and right child 
                    temp2 = da.get_at_index(current) 
                    da.set_at_index(current, da.get_at_index(right_child))
                    da.set_at_index(right_child, temp2)
                    
                    current = right_child 
                    left_child = (2 * current) + 1
                    right_child = (2 * current) + 2
                    
                # CASE 4: current element is in the right spot - exit while loop
                else:
                    left_child = da.length()
                    right_child = da.length()
                    
            current = temp - 1
        
        self.heap = da
        return None

class HashMap:
    def __init__(self, capacity: int, function) -> None:
        """
        Init new HashMap based on DA with SLL for collision resolution
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.buckets = DynamicArray()
        for _ in range(capacity):
            self.buckets.append(LinkedList())
        self.capacity = capacity
        self.hash_function = function
        self.size = 0

    def __str__(self) -> str:
        """
        Return content of hash map t in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        out = ''
        for i in range(self.buckets.length()):
            list = self.buckets.get_at_index(i)
            out += str(i) + ': ' + str(list) + '\n'
        return out

    def clear(self) -> None:
        """
        Method clears the contents of the hash map 
        """
        for index in range(0, self.buckets.length()):
            current_ll = self.buckets.get_at_index(index)
            if current_ll.head != None:
                current_ll.head = None 
                # current_ll.head.next = None
            self.size = 0
        return None 

    def get(self, key: str) -> object:
        """
        Method returns the value associated with the given key 
        """
        # Return value at the given key 
        hash_key = self.hash_function(key)
        
        index = hash_key % self.buckets.length()
        
        bucket = self.buckets.get_at_index(index)
        
        current = bucket.head
        while current != None:
            if current.key == key:
                return current.value 
            else:
                current = current.next
        return None 

    def put(self, key: str, value: object) -> None:
        """
        Method updates the key/value pair in the hash map
        """
        # 01. Get the correct key by running the given key through the hash function
        hash_key = self.hash_function(key)
        
        # 02. Get the length of the hash table 
        length = self.buckets.length()
        
        # 03. Get the correct index 
        index = hash_key % length 
        
        # Place value at correct key 
        bucket = self.buckets.get_at_index(index)
        if bucket.head == None:
            bucket.head = SLNode(key, value)
            self.size += 1
            return None
        else:
            current = bucket.head 
            previous = None
            # CASE 1: Look for duplicate keys and replace if found 
            while current != None:
                if current.key == key and previous == None:
                    temp = current.next 
                    bucket.head = SLNode(key, value)
                    bucket.head.next = temp
                    return None
                elif current.key == key:
                    temp = current.next 
                    current = SLNode(key, value)
                    current.next = temp 
                    previous.next = current 
                    return None
                previous = current
                current = current.next
            
            # CASE 2: If specified key is not a duplicate, insert it at head 
            current = bucket.head 
            bucket.head = SLNode(key, value)
            bucket.head.next = current 
            self.size += 1
            return None
                    
        return None

    def remove(self, key: str) -> None:
        """
        Method removes the given key and its associated value from the hash map
        """
        hash_key = self.hash_function(key)
        
        index = hash_key % self.buckets.length()
        
        bucket = self.buckets.get_at_index(index)
        
        if bucket.head != None:
            current = bucket.head 
            previous = None
            # CASE 1: Head bucket is matched key 
            if current.key == key and previous == None:
                bucket.head = current.next
                self.size -= 1
                return None
            # CASES: Any node except the last one 
            while current.next != None:
                if current.key == key:
                    previous.next = current.next
                    self.size -= 1
                    return None 
                    
                previous = current 
                current = current.next
            
            if current.key == key and previous != None:
                previous.next = None 
                self.size -= 1
                return None
            
            return None    

    def contains_key(self, key: str) -> bool:
        """
        Method returns True if given key is found in the hash table
        """
        hash_key = self.hash_function(key)
        
        index = hash_key % self.buckets.length()
        
        bucket = self.buckets.get_at_index(index)
        
        if bucket.head == None:
            return False 
        else:
            current = bucket.head 
            while current != None:
                if current.key == key:
                    return True 
                current = current.next
        
        return False

    def empty_buckets(self) -> int:
        """
        Method returns the number of empty buckets in the hash table 
        """
        count = 0
        for index in range(0, self.buckets.length()):
            bucket = self.buckets.get_at_index(index)
            if bucket.head == None:
                count += 1
        return count

    def table_load(self) -> float:
        """
        Method returns the current hash table load factor
        """
        elements = 0
        buckets = 0
        for index in range(0, self.buckets.length()):
            # Get the linked list stored in the bucket
            current_ll = self.buckets.get_at_index(index)
            # If linked list is not empty, count all the elements 
            if current_ll.head != None:
                node = current_ll.head
                while node != None:
                    elements += 1
                    node = node.next 
            # Count the buckets 
            buckets += 1
        
        load_factor = elements / buckets
        return load_factor
        
    def resize_table(self, new_capacity: int) -> None:
        """
        Method changes the capacity of the hash table 
        """
        
        # EXCEPTION 1: If new_capacity is less than 1, do nothing 
        if new_capacity < 1:
            return None
        
        # Get info from current hash table 
        old_keys = self.get_keys()
        
        # Create New Table 
        new_table = HashMap(new_capacity, self.hash_function)
        # Rehash old key-value pairs 
        for index in range(0, old_keys.length()):
            # Get old key
            current_key = old_keys.get_at_index(index)
            # Get value from old key
            current_value = self.get(current_key)
            
            # Now, store value at new index 
            hash = new_table.hash_function(current_key)
            new_index = hash % new_table.buckets.length()
            
            bucket = new_table.buckets.get_at_index(new_index)
            
            placed = 0
            if bucket.head == None:
                bucket.head = SLNode(current_key, current_value)
                placed = 1
                new_table.size += 1
            else:
                current = bucket.head
                previous = None 
                while current != None:
                    if current.key == current_key and previous == None:
                        temp = current.next
                        bucket.head = SLNode(current_key, current_value)
                        bucket.head.next = temp
                        placed = 1
                    elif current.key == current_key:
                        temp = current.next 
                        current = SLNode(current_key, current_value)
                        current.next = temp
                        previous.next = current 
                        placed = 1
                    
                    previous = current 
                    current = current.next 
                    
            if placed == 0:
                temp = bucket.head
                bucket.head = SLNode(current_key, current_value)
                bucket.head.next = temp
                new_table.size += 1
                
        # Set old table equal to new table 
        self.buckets = new_table.buckets 
        self.capacity = new_table.capacity 
        self.size = new_table.size    
        
        return None

    def get_keys(self) -> DynamicArray:
        """
        Method returns a DynamicArray that contains all the keys stored in the hash table
        """
        keys = DynamicArray()
        for index in range(0, self.buckets.length()):
            bucket = self.buckets.get_at_index(index)
            if bucket.head != None:
                current = bucket.head 
                while current != None:
                    keys.append(current.key)
                    current = current.next  
        return keys

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
        
    def length(self):
        return len(self._data)
        
    def get(self, index):
        return self._data[index]
        
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
        
class DirectedGraph:
    """
    Class to implement directed weighted graph
    - duplicate edges not allowed
    - loops not allowed
    - only positive edge weights
    - vertex names are integers
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency matrix
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.v_count = 0
        self.adj_matrix = []

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            v_count = 0
            for u, v, _ in start_edges:
                v_count = max(v_count, u, v)
            for _ in range(v_count + 1):
                self.add_vertex()
            for u, v, weight in start_edges:
                self.add_edge(u, v, weight)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        if self.v_count == 0:
            return 'EMPTY GRAPH\n'
        out = '   |'
        out += ' '.join(['{:2}'.format(i) for i in range(self.v_count)]) + '\n'
        out += '-' * (self.v_count * 3 + 3) + '\n'
        for i in range(self.v_count):
            row = self.adj_matrix[i]
            out += '{:2} |'.format(i)
            out += ' '.join(['{:2}'.format(w) for w in row]) + '\n'
        out = f"GRAPH ({self.v_count} vertices):\n{out}"
        return out

    # ------------------------------------------------------------------ #

    def add_vertex(self) -> int:
        """
        Method adds a new vertex to the graph
        """
        self.v_count = self.v_count + 1
        
        # append empty 
        self.adj_matrix.append([])
        
        for index1 in range(0, self.v_count):
            
            index2 = 0
            
            while index2 < self.v_count and len(self.adj_matrix[index1]) < self.v_count:
                self.adj_matrix[index1].append(0)
                
                index2 += 1
            
        return self.v_count
            
                        
        self.adj_matrix.append(list_to_append)
        
        return self.v_count

    def add_edge(self, src: int, dst: int, weight=1) -> None:
        """
        Method adds a new edge to a graph
        """
        # EDGE CASE 1: If src and dst are the same 
        if src == dst:
            return None
        
        # EDGE CASE 2: If weight is not positive 
        if weight < 0:
            return None
            
        # EDGE CASE 3: One or both vertices do not exist in graph 
        if src >= len(self.adj_matrix) or dst >= len(self.adj_matrix[src]):
            return None
        
        self.adj_matrix[src][dst] = weight
        return None

    def remove_edge(self, src: int, dst: int) -> None:
        """
        Method removes an edge from a graph
        """
        # EDGE CASE 1: If either or both vertices don't exist in graph 
        if src >= len(self.adj_matrix) or dst >= len(self.adj_matrix[src]):
            return None
            
        # EDGE CASE 1: If either or both vertices don't exist in graph 
        if src < 0 or dst < 0:
            return None
            
        self.adj_matrix[src][dst] = 0
        return None

    def get_vertices(self) -> []:
        """
        Method gets all the vertices in a graph 
        """
        vertices = []
        
        for index in range(0, self.v_count):
            
            vertices.append(index)
            
        return vertices

    def get_edges(self) -> []:
        """
        Method gets all the edges in a directed graph
        """
        edges = []
        
        for index1 in range(0, len(self.adj_matrix)):
            
            for index2 in range(0, len(self.adj_matrix[index1])):
                if self.adj_matrix[index1][index2] != 0:
                    edge = (index1, index2, self.adj_matrix[index1][index2])
                    edges.append(edge)
            
        return edges

    def is_valid_path(self, path: []) -> bool:
        """
        Method determines if a path is a valid path in a graph 
        """
        # EDGE CASE 1: An empty path is considered valid 
        if len(path) == 0:
            return True 
            
        for index in range(0, len(path) - 1):
            if self.adj_matrix[path[index]][path[index + 1]] == 0:
                return False 
                
        return True

    def dfs(self, v_start, v_end=None) -> []:
        """
        Method performs a dfs and returns a list of vertices visited
        """
        # EDGE CASE 1: v_start is not on graph 
        if v_start < 0 or v_start > len(self.adj_matrix):
            return []
            
        # Initialize an empty set of visited vertices 
        visited = set()
        # initialize an empty stack for processing 
        stack = Stack()
        # Initialize return list 
        return_list = []
        # Add first vertex to stack 
        stack.push(v_start)
        # While stack is not empty, repeat process 
        while stack.is_empty() == False:
            
            # pop a vertex 
            vertex = stack.pop()
            # If vertex is not in the set of visited vertices 
            if vertex not in visited:
                # Add vertex to set of visited vertices 
                visited.add(vertex)
                # Add vertex to return_list 
                return_list.append(vertex)
                # If you've hit v_end, end right there 
                if vertex == v_end:
                    return return_list 
                # push each vertex that is a direct successor of v to the stack 
                successors = []
                for index in range(0, len(self.adj_matrix[vertex])):
                    if self.adj_matrix[vertex][index] != 0:
                        successors.append(index)
                # Sort successors by value 
                for index in range(0, len(successors)):
                    while index > 0:
                        if successors[index] > successors[index - 1]:
                            successors[index], successors[index - 1] = successors[index - 1], successors[index]
                        else:
                            break
                        index -= 1
                # Push sorted successors into stack 
                for index in range(0, len(successors)):
                    stack.push(successors[index])
        return return_list

    def bfs(self, v_start, v_end=None) -> []:
        """
        Method performs a breadth-first search on a directed graph 
        """
        # EDGE CASE 1: v_start is not in graph 
        if v_start < 0 or v_start > len(self.adj_matrix): 
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
            vertex = queue.dequeue()
            
            # Only perform operations if vertex hasn't been visited yet
            if vertex not in visited:
                visited.add(vertex)
                return_list.append(vertex)
                    
                # If you've hit the v_end, end traversal:
                if vertex == v_end:
                    return return_list 
                        
                # Push direct successor into queue in alphabetical order 
                successors = []
                for index in range(0, len(self.adj_matrix[vertex])):
                    if self.adj_matrix[vertex][index] != 0:
                        # print("index: " + str(index))
                        successors.append(index)
                # Enqueue the successors into the queue 
                for index in range(0, len(successors)):
                    queue.enqueue(successors[index])
                        
        return return_list
        
    def detect_cycle(self, index, visited, recursion):
        """Helper function to determine if cycle in graph"""
        # Mark the current index as visited 
        visited[index] = True 
        # Add current index to recursion stack 
        recursion[index] = True
        
        for vertex in range(0, len(self.adj_matrix[index])):
            
            # If it's a neighbor 
            if self.adj_matrix[index][vertex] != 0:
                
                if visited[vertex] == False:
                    if self.detect_cycle(vertex, visited, recursion) == True:
                        return True
                elif recursion[vertex] == True:
                    return True 
        recursion[index] = False
        return False    

    def has_cycle(self):
        """
        Return True if graph contains a cycle, False otherwise
        """    
        
        # Initialize visited stack 
        visited = []
        for index in range(0, self.v_count + 1):
            visited.append(False)
        # Initialize recursion stack 
        recursion = []
        for index in range(0, self.v_count + 1):
            recursion.append(False)
        
        # Call the recursive function on all items in the graph 
        for index in range(0, self.v_count):
            
            # If vertex hasn't been visited yet, call helper function on it
            if visited[index] == False:
                answer = self.detect_cycle(index, visited, recursion)
                if answer == True:
                    return True 
        return False
                    
    def dijkstra(self, src: int) -> []:
        """
        Method implements the Dijkstra algorithm to compute the length of the shortest path from a given vertex to all other vertices in the graph
        """
        # Initialize the return list 
        return_list = []
        for index in range(0, len(self.adj_matrix)):
            return_list.append(float('inf'))
        
        # Initialize the hash table 
        hash = {}
        
        queue = MinHeap()
        
        current = [src, 0]
        queue.add(current)
        
        while queue.is_empty() == False:
            
            vertex = queue.remove_min()
            distance = vertex[1]
            
            if vertex[0] not in hash:
                
                hash[vertex[0]] = distance
                
                for index in range(0, len(self.adj_matrix[vertex[0]])):
                    if self.adj_matrix[vertex[0]][index] != 0:
                        new_distance = distance + self.adj_matrix[vertex[0]][index]
                        vertex_to_add = [index, new_distance]
                        queue.add(vertex_to_add)
                
        # Insert all the vertices in the hash into return_list 
        for index in range(0, len(self.adj_matrix)):
            if hash.get(index) != None:
                return_list[index] = (hash[index])
            else:
                return_list[index] = float('inf')
            
        return return_list

if __name__ == '__main__':

    # print("\nPDF - method add_vertex() / add_edge example 1")
    # print("----------------------------------------------")
    # g = DirectedGraph()
    # print(g)
    # for _ in range(5):
    #     g.add_vertex()
    # print(g)
    # 
    # edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
    #          (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    # for src, dst, weight in edges:
    #     g.add_edge(src, dst, weight)
    # print(g)
    # 
    # 
    # print("\nPDF - method get_edges() example 1")
    # print("----------------------------------")
    # g = DirectedGraph()
    # print(g.get_edges(), g.get_vertices(), sep='\n')
    # edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
    #          (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    # g = DirectedGraph(edges)
    # print(g.get_edges(), g.get_vertices(), sep='\n')
    # 
    
    # print("\nPDF - method is_valid_path() example 1")
    # print("--------------------------------------")
    # edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
    #          (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    # g = DirectedGraph(edges)
    # test_cases = [[0, 1, 4, 3], [1, 3, 2, 1], [0, 4], [4, 0], [], [2]]
    # for path in test_cases:
    #     print(path, g.is_valid_path(path))
    # 
    
    print("\nPDF - method dfs() and bfs() example 1")
    print("--------------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    for start in range(5):
        print(f'{start} DFS:{g.dfs(start)} BFS:{g.bfs(start)}')
    
    # 
    # print("\nPDF - method has_cycle() example 1")
    # print("----------------------------------")
    # edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
    #          (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    # g = DirectedGraph(edges)
    # 
    # edges_to_remove = [(3, 1), (4, 0), (3, 2)]
    # for src, dst in edges_to_remove:
    #     g.remove_edge(src, dst)
    #     print(g.get_edges(), g.has_cycle(), sep='\n')
    # 
    # edges_to_add = [(4, 3), (2, 3), (1, 3), (4, 0)]
    # for src, dst in edges_to_add:
    #     g.add_edge(src, dst)
    #     print(g.get_edges(), g.has_cycle(), sep='\n')
    # print('\n', g)
    # 
    # print("\nCustom Text - has_cycle")
    # print("-------------------------------")
    # 
    # edges = [(0, 7, 16), (2, 0, 18), (2, 1, 19), (2, 6, 19), (2, 11, 8), (3, 0, 11), (3, 2, 11), (3, 8, 7), (4, 1, 13), (6, 0, 20), (8, 10, 20), (10, 2, 5), (11, 3, 18)]
    # 
    # g = DirectedGraph(edges)
    # 
    # print(g.get_edges(), g.has_cycle(), sep="\n")
    
    # 
    # print("\nPDF - dijkstra() example 1")
    # print("--------------------------")
    # edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
    #          (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    # g = DirectedGraph(edges)
    # for i in range(5):
    #     print(f'DIJKSTRA {i} {g.dijkstra(i)}')
    # g.remove_edge(4, 3)
    # print('\n', g)
    # for i in range(5):
    #     print(f'DIJKSTRA {i} {g.dijkstra(i)}')
