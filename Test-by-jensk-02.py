# -*- coding: utf-8 -*-

'''
Created on 20.12.2023

@author: jens
'''

'''
1. Step: Define Simulation Input
    1.1. Load Network-Equipment
    1.2. Load Network-Topology
    1.3. Load Path-Request
2. Step: Define Simulation Settings #to do
3. Step: Run Simulation
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

import pprint


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

#create Di.Graph from python dictionary for network-topology using the provided equipment
network = network_from_json(network_topology_dict,nw_equipment)


# 1.3 Load Path-Request

# define path-request in python dict format
path_request_dict = get_from_storage_of_path_requests('example_02_linear') # 'example_01' or 'example_02_linear'

#create a path request using the path request dict and the equipment
path_request = create_path_request(nw_equipment, path_request_dict)
#print(type(path_request))
#pprint.pprint(path_request)
#print(path_request.__str__())
#print(path_request.__repr__())



#3. Simulate
number_of_simulation_example = 0

number_of_simulation_example = 2

if number_of_simulation_example == 1:
    #Simulation Example 1: using simulate function from Maksim
    path,infos = simulate(nw_equipment, network, path_request)
    print_path(path)
    x = final_gsnr(path)
    print('Final GSNR in dB')
    print(x)
elif number_of_simulation_example == 2:
    #Simulation Example 2: splitting the major simulate steps into 3 parts: 
    #1. building the network 
    #2. compute the path through the network
    #3. propagate the spectrum through the path in the network
    
    
    # show plot of network:
    nx.draw_planar(network, with_labels=True)
    plt.show()
    
    #1. build the network
    build_network_by_jensk(nw_equipment,network,path_request)
    # show plot of network:
    nx.draw_planar(network, with_labels=True)
    plt.show()
    
    #2. compute path through the network
    path = compute_constrained_path(network, path_request)
    
    pprint.pprint(path)
    #3. propagate the spectrum through the path in the network
    infos = propagate(path, path_request, nw_equipment)
    
    #4. Show Results 
    #4.1 path
    # pprint.pprint(path)
    print_path(path)
    #4.2 final snr
    x = final_gsnr(path)
    print('Final GSNR in dB')
    print(x)
    #4.3 history of si (Spectral information)
    final_si, history_of_si = get_history_of_si(path, path_request, nw_equipment)
    #print(history_of_si)
    #4.4 show plot of network:
    nx.draw_planar(network, with_labels=True)
    plt.show()
    
    #HISTORY of SI:
    array_for_x_axis=[]
    array_for_y_axis=[]
    # for i in range(0,len(history_of_si)):
    #     array_for_x_axis.append(i)
    #     array_for_y_axis.append(mean(history_of_si[i]._latency )*10**3)
    #
    # xpoints = np.array(array_for_x_axis)
    # ypoints = np.array(array_for_y_axis)
    # plt.plot(xpoints, ypoints, marker = 'o')
    # plt.xlabel("Number of element in chronological order") #or without *step_size -> Number of Iterations
    # plt.ylabel('Latency in ms')
    # plt.show()
    # for i in range(0,len(history_of_si)):
    #     array_for_x_axis.append(i)
    #     array_for_y_axis.append(mean(history_of_si[i]._nli))
    #
    # xpoints = np.array(array_for_x_axis)
    # ypoints = np.array(array_for_y_axis)
    # plt.plot(xpoints, ypoints, marker = 'o')
    # plt.xlabel("Number of element in chronological order") #or without *step_size -> Number of Iterations
    # plt.ylabel('nonlinear interference (NLI)')
    # plt.show()
    # from gnpy.core.utils import lin2db
    # for i in range(0,len(history_of_si)):
    #     array_for_x_axis.append(i)
    #     array_for_y_axis.append(lin2db(mean(history_of_si[i]._signal)))
    #
    # xpoints = np.array(array_for_x_axis)
    # ypoints = np.array(array_for_y_axis)
    # plt.plot(xpoints, ypoints, marker = 'o')
    # plt.xlabel("Number of element in chronological order") #or without *step_size -> Number of Iterations
    # plt.ylabel('Signal Power in db?')
    # plt.show()
 
    # for i in range(0,len(history_of_si)):
    #     array_for_x_axis.append(i)
    #     array_for_y_axis.append(mean(history_of_si[i]._chromatic_dispersion))
    #
    # xpoints = np.array(array_for_x_axis)
    # ypoints = np.array(array_for_y_axis)
    # plt.plot(xpoints, ypoints, marker = 'o')
    # plt.xlabel("Number of element in chronological order") #or without *step_size -> Number of Iterations
    # plt.ylabel('Chromatic dispersion in ps/nm')
    # plt.show()
    # for i in range(0,len(history_of_si)):
    #     array_for_x_axis.append(i)
    #     array_for_y_axis.append(mean(history_of_si[i]._pmd))
    #
    # xpoints = np.array(array_for_x_axis)
    # ypoints = np.array(array_for_y_axis)
    # plt.plot(xpoints, ypoints, marker = 'o')
    # plt.xlabel("Number of element in chronological order") #or without *step_size -> Number of Iterations
    # plt.ylabel('PMD? in ps')
    # plt.show()
    # for i in range(0,len(history_of_si)):
    #     array_for_x_axis.append(i)
    #     array_for_y_axis.append(mean(history_of_si[i]._pdl))
    #
    # xpoints = np.array(array_for_x_axis)
    # ypoints = np.array(array_for_y_axis)
    # plt.plot(xpoints, ypoints, marker = 'o')
    # plt.xlabel("Number of element in chronological order") #or without *step_size -> Number of Iterations
    # plt.ylabel('Polarization dependent loss (PDL) in db')
    # plt.show()
    
    # for i in range(0,len(history_of_si)):
    #     array_for_x_axis.append(i)
    #     array_for_y_axis.append(mean(history_of_si[i]._ase))
    #
    # xpoints = np.array(array_for_x_axis)
    # ypoints = np.array(array_for_y_axis)
    # plt.plot(xpoints, ypoints, marker = 'o')
    # plt.xlabel("Number of element in chronological order") #or without *step_size -> Number of Iterations
    # plt.ylabel('ase')
    # plt.show()
    
elif number_of_simulation_example == 3:
    # in this simulation the build network step was skipped: for testing and understanding purposes
    # does this simulation still make sense? I don't think so. Why?
    
    
    # show plot of network:
    nx.draw_planar(network, with_labels=True)
    plt.show()
    
    path = compute_constrained_path(network, path_request)
    infos = propagate(path, path_request, nw_equipment)
    print_path(path)
    x = final_gsnr(path)
    print('Final GSNR in dB')
    print(x)
    
    # show plot of network:
    nx.draw_planar(network, with_labels=True)
    plt.show()
    
    
#elif number_of_simulation_example == 4:
    
    
