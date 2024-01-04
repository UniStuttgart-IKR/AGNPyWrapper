'''
Created on 03.01.2024

@author: jens
'''

#import important functions
from gnpy.tools.json_io import _equipment_from_json, network_from_json
from gnpy.topology.request import compute_constrained_path, propagate
from GNPyWrapper import create_path_request, simulate, print_path, final_gsnr, build_network_by_jensk, get_history_of_si
from GNPyWrapper import get_from_storage_of_network_equipment
from GNPyWrapper import get_from_storage_of_network_topologies
from GNPyWrapper import get_from_storage_of_path_requests

import networkx as nx
import matplotlib.pyplot as plt

import numpy as np
from numpy import mean

#needed for network from json:
from networkx import DiGraph

from gnpy.tools.json_io import _cls_for, merge_equalization
from gnpy.core.utils import merge_amplifier_restrictions
from gnpy.core import elements

# for better printing
import pprint


# get json_data 
# -> use the function

# get the equipment
# -> use the function


#1. Step: Define Simulation Input

#1.1 Load Network-Equipment

#Network-equipment in python dictionary format
network_equipment_dict = get_from_storage_of_network_equipment('example_01_default')     

#create a random filename
filename = 'random-filename' # this is necessary for Amp.from_json function...

# do gnpy stuff with it...
nw_equipment= _equipment_from_json(network_equipment_dict, filename)


#1.2 Load Network topology
# network-topology-dict
network_topology_dict= get_from_storage_of_network_topologies('example_02_linear') # 'example_01' or 'example_02_linear'


simulation = 0

simulation = 3

if simulation == 1:
    #def network_from_json(json_data, equipment):
        # NOTE|dutc: we could use the following, but it would tie our data format
        #            too closely to the graph library
        # from networkx import node_link_graph
    g = DiGraph()
    for el_config in network_topology_dict['elements']:
        typ = el_config.pop('type')
        variety = el_config.pop('type_variety', 'default')
        cls = _cls_for(typ)
        if typ == 'Fused':
            # well, there's no variety for the 'Fused' node type
            pass
        elif variety in nw_equipment[typ]:
            extra_params = nw_equipment[typ][variety].__dict__
            temp = el_config.setdefault('params', {})
            if typ == 'Roadm':
                # if equalization is defined, remove default equalization from the extra_params
                # If equalisation is not defined in the element config, then use the default one from equipment
                # if more than one equalization was defined in element config, then raise an error
                extra_params = merge_equalization(temp, extra_params)
                if not extra_params:
                    msg = f'ROADM {el_config["uid"]}: invalid equalization settings'
                    raise ConfigurationError(msg)
            temp = merge_amplifier_restrictions(temp, extra_params)
            el_config['params'] = temp
            el_config['type_variety'] = variety
        elif (typ in ['Fiber', 'RamanFiber']):
            raise ConfigurationError(f'The {typ} of variety type {variety} was not recognized:'
                                     '\nplease check it is properly defined in the eqpt_config json file')
        elif typ == 'Edfa':
            if variety in ['default', '']:
                el_config['params'] = Amp.default_values
            else:
                raise ConfigurationError(f'The Edfa of variety type {variety} was not recognized:'
                                         '\nplease check it is properly defined in the eqpt_config json file')
        el = cls(**el_config)
        g.add_node(el)
    
    nodes = {k.uid: k for k in g.nodes()}
    
    for cx in network_topology_dict['connections']:
        from_node, to_node = cx['from_node'], cx['to_node']
        try:
            if isinstance(nodes[from_node], elements.Fiber):
                edge_length = nodes[from_node].params.length
            else:
                edge_length = 0.01
            g.add_edge(nodes[from_node], nodes[to_node], weight=edge_length)
        except KeyError:
            msg = f'can not find {from_node} or {to_node} defined in {cx}'
            raise NetworkTopologyError(msg)
    
    #    return g
    print(nodes)
    print(g.edges())
    #pprint.pprint(g.edges())
    print(g)
    
    # nx.draw_circular(g, with_labels=True)
    nx.draw_planar(g, with_labels=True)
    plt.show()
elif simulation == 2:
    # Analyse von network_from_json function
    g = DiGraph()
    for el_config in network_topology_dict['elements']:
        # print(el_config)
        typ = el_config.pop('type')
        print('typ: ' + typ)
        variety = el_config.pop('type_variety', 'default') 
        print('variety: ' + variety)
        cls = _cls_for(typ) # cls stands for class of tpy
        print('cls: ' + str(cls))
        
        
elif simulation == 3:
    g = DiGraph()
    for el_config in network_topology_dict['elements']:
        typ = el_config.pop('type')
        variety = el_config.pop('type_variety', 'default')
        cls = _cls_for(typ)
        
        # added by jensk
        print('0')
        print(el_config)
        print('1')
        print('typ: ' + typ)
        print('variety: ' + variety)
        print('cls: ' + str(cls))
        # end of added code
        
        
        if typ == 'Fused':
            # well, there's no variety for the 'Fused' node type
            pass
        elif variety in nw_equipment[typ]:
            extra_params = nw_equipment[typ][variety].__dict__
            print('2')
            print(extra_params)
            temp = el_config.setdefault('params', {})
            print('3')
            print(temp)
            if typ == 'Roadm':
                # if equalization is defined, remove default equalization from the extra_params
                # If equalisation is not defined in the element config, then use the default one from equipment
                # if more than one equalization was defined in element config, then raise an error
                extra_params = merge_equalization(temp, extra_params)
                if not extra_params:
                    msg = f'ROADM {el_config["uid"]}: invalid equalization settings'
                    raise ConfigurationError(msg)
            temp = merge_amplifier_restrictions(temp, extra_params)
            print('4')
            print(temp)
            el_config['params'] = temp
            el_config['type_variety'] = variety
            
            # added by jensk
            print('5')
            print('typ: ' + typ)
            print('variety: ' + variety)
            print('cls: ' + str(cls))
            # end of added code
            print('6')
            print(el_config)

        elif (typ in ['Fiber', 'RamanFiber']):
            raise ConfigurationError(f'The {typ} of variety type {variety} was not recognized:'
                                     '\nplease check it is properly defined in the eqpt_config json file')
            

        elif typ == 'Edfa':
            if variety in ['default', '']:
                el_config['params'] = Amp.default_values
            else:
                raise ConfigurationError(f'The Edfa of variety type {variety} was not recognized:'
                                         '\nplease check it is properly defined in the eqpt_config json file')


        el = cls(**el_config)
        g.add_node(el)
        
        nx.draw_planar(g, with_labels=True)
        plt.show()
        
    nx.draw_planar(g, with_labels=True)
    plt.show()
    
    nodes = {k.uid: k for k in g.nodes()}
    
    for cx in network_topology_dict['connections']:
        from_node, to_node = cx['from_node'], cx['to_node']
        try:
            if isinstance(nodes[from_node], elements.Fiber):
                edge_length = nodes[from_node].params.length
            else:
                edge_length = 0.01
            g.add_edge(nodes[from_node], nodes[to_node], weight=edge_length)
        except KeyError:
            msg = f'can not find {from_node} or {to_node} defined in {cx}'
            raise NetworkTopologyError(msg)
        nx.draw_planar(g, with_labels=True)
        plt.show()
    
    nx.draw_planar(g, with_labels=True)
    plt.show()
    
elif simulation == 4:
    