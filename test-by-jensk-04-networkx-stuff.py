'Edfa_2''''
Created on 03.01.2024

@author: jens
'''
import networkx as nx
import matplotlib.pyplot as plt

from networkx import (dijkstra_path, NetworkXNoPath,
                      all_simple_paths, shortest_simple_paths)

simulation = 0

simulation = 7

if simulation == 1:
    G = nx.Graph()
    G.add_edge(1,2)
    G.add_edge(2,3)
    G.add_edge(3,4)
    G.add_edge(4,1)
    print(G.edges())
    print(G.nodes())
    print(1 in G)
    nx.draw_circular(G, with_labels=True)
    plt.show()
elif simulation == 2:
    g_parent = nx.Graph()
    g_parent.add_edge("component_1","component_2")
    nx.draw_planar(g_parent, with_labels=True)
    plt.show()
    
    component_1_dict = {}
    
    
    g_child = nx.Graph()
    
elif simulation ==3:
    g_parent = nx.Graph()
    

    
    g_child_1 = nx.Graph()
    g_child_2 = nx.Graph()  
    
    #g_child_1['graph'] = nx.Graph()
    #g_child_2['graph'] = nx.Graph()
    
    #g_child_1.add_edge(child_1, child_2)
    
    g_parent.add_edge(g_child_1,g_child_2)
    
    
    
    
    nx.draw_planar(g_parent, with_labels=True)
    plt.show()
    
elif simulation == 4:
    g_parent = nx.Graph()
    
    g_child_1={}
    g_child_2={}
    
    g_child_1['name'] = 'child_1'
    g_child_2['name'] = 'child_2'  
    
    g_child_1['graph'] = nx.Graph()
    g_child_2['graph'] = nx.Graph()
    
    #g_child_1.add_edge(child_1, child_2)
    
    g_parent.add_edge(g_child_1['name'],g_child_2['name'])
    
    
    
    
    nx.draw_planar(g_parent, with_labels=True)
    plt.show()
elif simulation == 5:
    g_parent = nx.DiGraph()
    
    g_child_1={}
    g_child_2={}
    
    g_child_1['name'] = 'child_1'
    g_child_2['name'] = 'child_2'  
    
    g_child_1['graph'] = nx.DiGraph()
    g_child_2['graph'] = nx.DiGraph()
    
    g_child_1['graph'].add_edge(1, 2)
    g_child_1['graph'].add_edge(2, 3)
    g_child_1['graph'].add_edge(3, 4)
    g_child_1['graph'].add_edge(4, 1)
    
    g_parent.add_edge(g_child_1['name'],g_child_2['name'])
    
    
    
    
    nx.draw_planar(g_parent, with_labels=True)
    plt.show()
    
    nx.draw_planar(g_child_1['graph'], with_labels=True)
    plt.show()
elif simulation == 6:
    g_parent = nx.DiGraph()
    
    g_child_1={}
    g_child_2={}
    
    g_child_1['name'] = 'child_1'
    g_child_2['name'] = 'child_2'  
    
    g_child_1['graph'] = nx.DiGraph()
    g_child_2['graph'] = nx.DiGraph()
    
    g_child_1['graph'].add_edge(1, 2)
    g_child_1['graph'].add_edge(2, 3)
    g_child_1['graph'].add_edge(3, 4)
    g_child_1['graph'].add_edge(4, 1)
    
    g_parent.add_edge(g_child_1['name'],g_child_2['name'])
    
    g_child_2['graph'].add_edge('a', 'b')
    g_child_2['graph'].add_edge('b', 'c')
    g_child_2['graph'].add_edge('c', 'd')
    g_child_2['graph'].add_edge('d', 'a')
    
    
    
    nx.draw_planar(g_parent, with_labels=True)
    plt.show()
    
    nx.draw_planar(g_child_1['graph'], with_labels=True)
    plt.show()
    
    nx.draw_planar(g_child_2['graph'], with_labels=True)
    plt.show()
    
    
    F = nx.compose(g_child_1['graph'], g_child_2['graph']) # compose is very important
    
    nx.draw_planar(F, with_labels=True)
    plt.show()
    
    F.add_edge(4,'a')
    
    nx.draw_planar(F, with_labels=True)
    plt.show()
    

    # print(g_child_1)
    
    
    
    # add path computation:
    
    path_generator= shortest_simple_paths(F,3,'c')
    #total_path = next(path for path in path_generator)
    paths = []
    for path in path_generator:
        paths.append(path)
    print(path_generator)
    print(paths)
    #print(total_path)
    
    
    
    F.add_edge(3,'c')
    F.add_edge(3,'b')
    nx.draw_planar(F, with_labels=True)
    plt.show()   
    
    path_generator= shortest_simple_paths(F,3,'c')
    print(path_generator)
    # total_path_02 = next(path for path in path_generator)
    # print(total_path_02)
    paths = []
    for path in path_generator:
        paths.append(path)
    print(paths)
elif simulation == 7:
    
    # display overall graph(/network) idea:
    Abstract_Graph= {}
    Abstract_Graph['name']= 'Overview of network'    
    Abstract_Graph['graph'] = nx.DiGraph()
    
    network_component_01 = {}
    network_component_02 = {}
    network_component_03 = {}
    
    network_component_01['name'] = 'City A'
    network_component_02['name'] = 'Link between City A and B'
    network_component_03['name'] = 'City B'
    
    Abstract_Graph['graph'].add_edge(network_component_01['name'],network_component_02['name'] )
    Abstract_Graph['graph'].add_edge(network_component_02['name'],network_component_03['name'] )
    
    nx.draw_planar(Abstract_Graph['graph'], with_labels=True)
    plt.show() 
    
    
    # define different network components
    network_component_01['graph'] = nx.DiGraph()
    network_component_02['graph'] = nx.DiGraph()
    network_component_03['graph'] = nx.DiGraph()
    
    #add network elements to component 1
    
    network_component_01['graph'].add_edge('Transceiver_A', 'Roadm_A')
    
    nx.draw_planar(network_component_01['graph'], with_labels=True)
    plt.show() 
    #add network elements to component 2
    #create list of edges
    elist_01 = [('Fiber_1','Edfa_1'), ('Edfa_1','Fiber_2'),('Fiber_2','Edfa_2'), ('Edfa_2','Fiber_3'),('Fiber_3','Edfa_3'), ('Edfa_3','Fiber_4')]
    
    network_component_02['graph'].add_edges_from(elist_01)
    nx.draw_planar(network_component_02['graph'], with_labels=True)
    plt.show() 
    
    
    #add network elements to component 3
    
    network_component_03['graph'].add_edge('Roadm_B', 'Transceiver_B')
    
    nx.draw_planar(network_component_03['graph'], with_labels=True)
    plt.show() 
    
    # compose component 1 and 2
    
    F = nx.compose(network_component_01['graph'], network_component_02['graph'])
    nx.draw_planar(F, with_labels=True)
    plt.show() 
    
    # compose component 1,2 with 3
    F_2 = nx.compose(F,network_component_03['graph'])
    nx.draw_planar(F_2, with_labels=True)
    plt.show()
    
    
    # connect in graph
    F_2.add_edge('Roadm_A', 'Fiber_1')
    nx.draw_planar(F_2, with_labels=True)
    plt.show()
    # second connection
    F_2.add_edge('Fiber_4', 'Roadm_B')
    nx.draw_planar(F_2, with_labels=True)
    plt.show()
    
    path_generator= shortest_simple_paths(F_2,'Transceiver_A','Transceiver_B')
    print(path_generator)
    # total_path_02 = next(path for path in path_generator)
    # print(total_path_02)
    paths = []
    for path in path_generator:
        paths.append(path)
    print(paths)
    
    #through this path later the signal gets propagated
    
    # add another component
    network_component_04 = {}
    network_component_04['name'] = 'Shorter_Fiber'
    network_component_04['graph'] = nx.DiGraph()
    
    # add to abstract graph
    Abstract_Graph['graph'].add_edge(network_component_01['name'],network_component_04['name'] )
    Abstract_Graph['graph'].add_edge(network_component_04['name'],network_component_03['name'] )
    nx.draw_planar(Abstract_Graph['graph'], with_labels=True)
    plt.show()     

    
    #create list of edges

    
    elist_02 = [('Fiber_a','Edfa_a'), ('Edfa_a','Fiber_b')]
    
    network_component_04['graph'].add_edges_from(elist_02)
    nx.draw_planar(network_component_04['graph'], with_labels=True)
    plt.show() 
    
    # add (with compose function) the new network element
    F_2 = nx.compose(F_2,network_component_04['graph'])
    nx.draw_planar(F_2, with_labels=True)
    plt.show() 
    
    # connect new component with current network
    F_2.add_edge('Roadm_A', 'Fiber_a')
    nx.draw_planar(F_2, with_labels=True)
    plt.show()
    # second connection
    F_2.add_edge('Fiber_b', 'Roadm_B')
    nx.draw_planar(F_2, with_labels=True)
    plt.show()
    
    
    path_generator= shortest_simple_paths(F_2,'Transceiver_A','Transceiver_B')
    print(type(path_generator))
    print(path_generator)
    # total_path_02 = next(path for path in path_generator)
    # print(total_path_02)
    paths = []
    for path in path_generator:
        paths.append(path)
    print(paths)
    
    path_generator_02= shortest_simple_paths(F_2,'Transceiver_A','Transceiver_B')
    total_path_02 = next(path for path in path_generator_02)
    print(total_path_02)
    