# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 11:49:07 2023

@author: Maksims Zabetcuks
"""

import sys
#sys.path.append('C:\\Users\\jens\\anaconda3\envs\\forschungsarbeit_env\\oopt-gnpy')
import gnpy

#sys.path.append('C:\\Users\\jens\\anaconda3\envs\\forschungsarbeit_env\\AGNPyWrapper\\GNPyWrapper')



from GNPyWrapper import (load_data, load_net, load_eqpt, create_path_request, customize_si, simulate, final_gsnr, customize_amp, 
                                customize_all_fiber, customize_roadm, customize_span, customize_one_fiber, print_path, sim_reach)
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from numpy import mean

from gnpy.tools.json_io import load_json, _equipment_from_json,\
    network_from_json

from pathlib import Path

#Params:
eqpt_path = 'eqpt_config_IKR_wrap.json'
topology_path = 'meshTopologyExampleV2.json'
lin_topology_path = 'LinearTopology.json'
#Use 1:
# equipment, network = load_data(eqpt_path, topology_path)
#Use 2:
equipment = load_eqpt(eqpt_path)
network = load_net(topology_path, equipment)

# of which type is the (network) equipment?
#print(type(equipment))
import pprint
#pprint.pprint(equipment)
# of which type is the network?
#print(type(network))

#test network stuff
#print("Lannion_CAS" in network)

"""TESTING SI"""
# #Params:
# si_parameters = { 
#         "f_min": 191.3e12,
#         "f_max": 196.1e12,
#         "baud_rate": 32e9,
#         "spacing": 50e9,
#         "power_dbm": 0,
#         "power_range_db": [0, 0, 0.5],
#         "roll_off": 0.15,
#         "tx_osnr": 20,
#         "sys_margins": 0
#         }
# #Use:
# customize_si(equipment, si_parameters) 
"""WORKS"""


"""TESTING FIBER"""
#Params:
fiber_parameters = { #SSMF
    'length': 90,
    'length_units': 'km',
    'dispersion': 1.67e-05,
    'effective_area': 83e-12,
    'pmd_coef': 1.265e-15,
    'loss_coef': 0.2
    #'ref_wavelength' OR 'ref_frequency' can also be specified
}
# fiber_uid = 'fiber (Quimper → Lorient_KMA)-'
# #Use:
# customize_all_fiber(network, fiber_parameters)
# # customize_one_fiber(network, fiber_uid, fiber_parameters)

"""WORKS"""

"""TESTING SPAN"""
# # Params:
# span_parameters = {
#       'power_mode': True,
#       'delta_power_range_db': [-2,3,0.5],
#       'max_fiber_lineic_loss_for_raman': 0.25,
#       'target_extended_gain': 2.5,
#       'max_length': 50,  
#       'length_units': 'km',
#       'max_loss': 28,
#       'padding': 10,
#       'EOL': 0,
#       'con_in': 0,
#       'con_out': 0
#   }
# #Use:
# customize_span(equipment, span_parameters)
"""WORKS"""

"""TESTING ROADM"""
# # Params:
# roadm_parameters = {
#     'target_pch_out_db': -20,            
#     'add_drop_osnr': 33,
#     'pmd': 3e-12,
#     'pdl': 1.5,
#     'restrictions': {
#         'preamp_variety_list': [],
#         'booster_variety_list': []
#     }
# }
# #Use:
# customize_roadm(equipment, roadm_parameters)
"""WORKS"""

"""TESTING AMP"""
# #Params:
# Model_vg = namedtuple('Model_vg', 'nf1 nf2 delta_p orig_nf_min orig_nf_max')
# Model_fg = namedtuple('Model_fg', 'nf0')
# Model_openroadm_ila = namedtuple('Model_openroadm_ila', 'nf_coef')
# Model_hybrid = namedtuple('Model_hybrid', 'nf_ram gain_ram edfa_variety')
# Model_dual_stage = namedtuple('Model_dual_stage', 'preamp_variety booster_variety')

# amplifier_parameters = {
#     'f_min': 191.35e12,
#     'f_max': 196.1e12,
#     'type_variety': '', 
#     'type_def': 'openroadm', 
#     'gain_flatmax': 27,
#     'gain_min': 0,
#     'p_max': 22,
#     'nf_model': Model_openroadm_ila([-0.0008104, -0.06221, -0.5889, 37.62]), 
#     'dual_stage_model': None,
#     'nf_fit_coeff': None,
#     'nf_ripple': None, 
#     'dgt': [0], 
#     'gain_ripple': None, 
#     'out_voa_auto': False,
#     'allowed_for_design': True, 
#     'raman': False,
#     'pmd': 0,
#     'pdl': 0
# }
# #Use:
# customize_amp(equipment, amplifier_parameters)
"""WORKS"""

"""ALL TOGETHER WORK AS WELL"""
         

# Params:
req_params_lin = {
    'request_id': 0,
    'trx_type': 'OpenZR+', 
    'trx_mode': '100ZR+, DP-QPSK',
    'source': 'trx_source', 
    'destination': 'trx_destination', 
    'bidir': False,
    'nodes_list': ['trx_destination'], 
    'loose_list': ['strict'],
    'format': '100 Gbit/s, DP-QPSK',
    'path_bandwidth': 0,
    'effective_freq_slot': None,
    'baud_rate': 34170000000.0, 
    'OSNR': 10.5, 
    'bit_rate': 100000000000.0, 
    'roll_off': 0.15, 
    'tx_osnr': 36, 
    'min_spacing': 50000000000.0, 
    'cost': 1, 
    'penalties': {}, 
    'f_min': 191.3e12, 
    'f_max': 195.1e12, 
    'power': 0.001, 
    'spacing': None
    } 

req_params_mesh = {
    'request_id': 0,
    'trx_type': 'OpenZR+', 
    'trx_mode': '300ZR+, DP-8QAM',
    'source': 'trx Brest_KLA', 
    'destination': 'trx Vannes_KBE', 
    'bidir': False,
    'nodes_list': ['trx Vannes_KBE'], 
    'loose_list': ['strict'],
    'format': '100 Gbit/s, DP-QPSK',
    'path_bandwidth': 0,
    'effective_freq_slot': None,
    'baud_rate': 34170000000.0, 
    'OSNR': 10.5, 
    'bit_rate': 100000000000.0, 
    'roll_off': 0.15, 
    'tx_osnr': 36, 
    'min_spacing': 50000000000.0, 
    'cost': 1, 
    'penalties': {}, 
    'f_min': 191.3e12, 
    'f_max': 195.1e12, 
    'power': 0.001, 
    'spacing': None
    }


#print(type(req_params_lin))
#print(type(req_params_mesh))

'''test simulations-to better understand gnpy/GnpyWrapper'''
number_of_simulation = 0
#state which simulation you want to run
number_of_simulation = 1002

if number_of_simulation == 1:
    #test of simulate function
    #Use:   
    path_request = create_path_request(equipment, req_params_mesh)
    print("This is the type of the path_request:")
    print(type(path_request))
    print(path_request.initial_spectrum)
    #Use:
    path,infos, infos_02, si_list_02, infos_03, si_list_03 = simulate(equipment, network, path_request)
    
    print("Das ist die Länge der SI Liste:")
    print(len(si_list_02))
    print("typ eines elements der liste")
    print(type(si_list_02[1]))
    pprint.pprint(si_list_02)
    for i in range(0,len(si_list_02)):
        print(i)
        print(mean(si_list_02[i]._nli))
    
    print(type(path))
    print(len(path))
    # print(path[5])
    print(path[5]._psig_in)
    print_path(path)
    x = final_gsnr(path)
    print('GSNR at the beginning')
    print('TODO')

    print('Final GSNR in dB')
    print(x)
    print('GSNR/OSNR? requirenment at the end:')
    print(path_request.OSNR)
    # print('info output of simulate function:')
    # print(type(infos.frequency))
    # print(infos.frequency)
    # print(len(infos.frequency))
    # print(infos.tx_osnr)
    # print(len(infos.tx_osnr))
    # print(infos.signal)
    # print(len(infos.signal))
    ## Tested
    # plt.plot(infos.frequency, infos.signal, marker = 'o')
    # plt.xlabel("Frequency") #or without *step_size -> Number of Iterations
    # plt.ylabel("signal")
    # plt.show()
    
    print(path[-1].snr_01nm)
    print(path[-1].propagated_labels)
    print(type(infos))
    print(infos._tx_osnr)
    print(infos._chromatic_dispersion)
    print(infos._pmd)
    print(type(infos._pref))
    print(infos._pref)
    print(type(infos._signal))
    print(infos._signal)
    print(type(infos._delta_pdb_per_channel))
    print(infos._delta_pdb_per_channel)
    print(mean(path[-1].snr_01nm))
    print(mean(infos._latency))
    print(infos)
    print(type(si_list_03))
    print(type(si_list_03[1]))
    for i in range(0,len(si_list_03)):
        print(i)
        print(mean(si_list_03[i]._frequency))
    array_for_x_axis=[]
    array_for_y_axis=[]
    # for i in range(0,len(si_list_03)):
    #     array_for_x_axis.append(i)
    #     array_for_y_axis.append(mean(si_list_03[i]._latency )*10**3)
    #
    # xpoints = np.array(array_for_x_axis)
    # ypoints = np.array(array_for_y_axis)
    # plt.plot(xpoints, ypoints, marker = 'o')
    # plt.xlabel("Number of element in chronological order") #or without *step_size -> Number of Iterations
    # plt.ylabel('Latency in ms')
    # plt.show()
    # for i in range(0,len(si_list_03)):
    #     array_for_x_axis.append(i)
    #     array_for_y_axis.append(mean(si_list_03[i]._nli))
    #
    # xpoints = np.array(array_for_x_axis)
    # ypoints = np.array(array_for_y_axis)
    # plt.plot(xpoints, ypoints, marker = 'o')
    # plt.xlabel("Number of element in chronological order") #or without *step_size -> Number of Iterations
    # plt.ylabel('nonlinear interference (NLI)')
    # plt.show()
    # from gnpy.core.utils import lin2db
    # for i in range(0,len(si_list_03)):
    #     array_for_x_axis.append(i)
    #     array_for_y_axis.append(lin2db(mean(si_list_03[i]._signal)))
    #
    # xpoints = np.array(array_for_x_axis)
    # ypoints = np.array(array_for_y_axis)
    # plt.plot(xpoints, ypoints, marker = 'o')
    # plt.xlabel("Number of element in chronological order") #or without *step_size -> Number of Iterations
    # plt.ylabel('Signal Power in db?')
    # plt.show()
 
    # for i in range(0,len(si_list_03)):
    #     array_for_x_axis.append(i)
    #     array_for_y_axis.append(mean(si_list_03[i]._chromatic_dispersion))
    #
    # xpoints = np.array(array_for_x_axis)
    # ypoints = np.array(array_for_y_axis)
    # plt.plot(xpoints, ypoints, marker = 'o')
    # plt.xlabel("Number of element in chronological order") #or without *step_size -> Number of Iterations
    # plt.ylabel('Chromatic dispersion in ps/nm')
    # plt.show()
    # for i in range(0,len(si_list_03)):
    #     array_for_x_axis.append(i)
    #     array_for_y_axis.append(mean(si_list_03[i]._pmd))
    #
    # xpoints = np.array(array_for_x_axis)
    # ypoints = np.array(array_for_y_axis)
    # plt.plot(xpoints, ypoints, marker = 'o')
    # plt.xlabel("Number of element in chronological order") #or without *step_size -> Number of Iterations
    # plt.ylabel('PMD? in ps')
    # plt.show()
    # for i in range(0,len(si_list_03)):
    #     array_for_x_axis.append(i)
    #     array_for_y_axis.append(mean(si_list_03[i]._pdl))
    #
    # xpoints = np.array(array_for_x_axis)
    # ypoints = np.array(array_for_y_axis)
    # plt.plot(xpoints, ypoints, marker = 'o')
    # plt.xlabel("Number of element in chronological order") #or without *step_size -> Number of Iterations
    # plt.ylabel('Polarization dependent loss (PDL) in db')
    # plt.show()
    
    # for i in range(0,len(si_list_03)):
    #     array_for_x_axis.append(i)
    #     array_for_y_axis.append(mean(si_list_03[i]._ase))
    #
    # xpoints = np.array(array_for_x_axis)
    # ypoints = np.array(array_for_y_axis)
    # plt.plot(xpoints, ypoints, marker = 'o')
    # plt.xlabel("Number of element in chronological order") #or without *step_size -> Number of Iterations
    # plt.ylabel('ase')
    # plt.show()
    

        
elif number_of_simulation == 2:
    #second simulation using different topology
    #test of use case example of GNPyWrapper using sim_reach function
    print('Start of Second Simulation using different topology')
    step_size = 90
    fin_gsnr, fib_length,counter,path_02, list_gsnr,network_sim_02,p_r_02 = sim_reach(eqpt_path, lin_topology_path, req_params_lin, fiber_parameters, step_size)
    print('Path of Second Simulation:')
    print_path(path_02)
    print('Final GSNR ')
    print(fin_gsnr)
    print('GSNR/OSNR requirenment:')
    print('TODO')
    print('Final fiber length in km: ')
    print(fib_length)
    print('number of iterations')
    print(counter)
    print('gsnr development')
    print(list_gsnr)
    print("Path Request OSNR:")
    print(p_r_02.OSNR)
    print(network_sim_02)
    print(type(network_sim_02))
    print(len(network_sim_02))
    print(1 in network_sim_02)
    print("trx_destination" in network_sim_02)
    # print(network_sim_02.edges()) # i don't know what this prints...
    # i= 0
    # while i <= counter:
    #     print()
    # import matplotlib.pyplot as plt
    # import numpy as np
    
    # xpoints = np.array([1, counter])
    # ypoints = list_gsnr
    array_for_x_axis =[]
    i= 1
    while i <= counter:
        i = i + 1
        array_for_x_axis.append(i*step_size) #or without *step_size
    xpoints = np.array(array_for_x_axis)
    ypoints = np.array(list_gsnr)
    plt.plot(xpoints, ypoints, marker = 'o')
    plt.xlabel("Fiber length in km") #or without *step_size -> Number of Iterations
    plt.ylabel("GSNR")
    plt.show()
    # visualize final network
    import networkx as nx
    nx.draw_spectral(network_sim_02, with_labels=True)
    plt.show()

elif number_of_simulation == 3:
    print('Edfa east edfa in Vannes_KBE to Lorient_KMA' in network)
    import networkx as nx
    nx.draw_planar(network, with_labels=True)
    plt.show()

elif number_of_simulation == 4:
    import networkx as nx
    nx.draw_circular(network, with_labels=True)
    plt.show()
elif number_of_simulation == 5:
    print(dict(network.degree)["trx Brest_KLA"]) # fails
elif number_of_simulation == 6:
    # simulation description: tests if the normal networkx functions are callable on  network 
    # print(list(network.nodes))
    # abab= [3,4,5]
    #
    # print(abab)
    # print(network.nodes()) #fails
    # print(network.edges())
    network.nodes()
    #print(network.nodes())
elif number_of_simulation == 101:
    # simulation description: Tests the functionalities of the networkx package
    # abcd = 4 # random code
    import networkx as nx
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
elif number_of_simulation == 1001:
    print('The following simulation is running:' + str(number_of_simulation))
    #from pathlib import Path
    json_data= load_json(Path(eqpt_path))
    print(type(json_data))
    #print(json_data)
    #pprint.pprint(json_data)
    json_data_03= {'Edfa': [{'allowed_for_design': False,
           'gain_flatmax': 26,
           'gain_min': 15,
           'nf_max': 10,
           'nf_min': 6,
           'out_voa_auto': False,
           'p_max': 23,
           'type_def': 'variable_gain',
           'type_variety': 'operator_model_example'},
          {'allowed_for_design': False,
           'gain_flatmax': 27,
           'gain_min': 0,
           'nf_coef': [-0.0008104, -0.06221, -0.5889, 37.62],
           'p_max': 22,
           'type_def': 'openroadm',
           'type_variety': 'openroadm_ila_low_noise'},
          {'allowed_for_design': True,
           'gain_flatmax': 27,
           'gain_min': 0,
           'nf_coef': [-0.0005952, -0.0625, -1.071, 28.99],
           'p_max': 22,
           'type_def': 'openroadm',
           'type_variety': 'openroadm_ila_standard'},
          {'allowed_for_design': False,
           'gain_flatmax': 27,
           'gain_min': 0,
           'p_max': 22,
           'type_def': 'openroadm_preamp',
           'type_variety': 'openroadm_mw_mw_preamp'},
          {'allowed_for_design': False,
           'gain_flatmax': 27,
           'gain_min': 0,
           'nf_coef': [-0.0005952, -0.0625, -1.071, 28.99],
           'p_max': 22,
           'type_def': 'openroadm',
           'type_variety': 'openroadm_mw_mw_preamp_typical_ver5'},
          {'allowed_for_design': False,
           'gain_flatmax': 27,
           'gain_min': 0,
           'nf_coef': [-0.0005952, -0.0625, -1.071, 27.99],
           'p_max': 22,
           'pdl': 0,
           'pmd': 0,
           'type_def': 'openroadm',
           'type_variety': 'openroadm_mw_mw_preamp_worstcase_ver5'},
          {'allowed_for_design': False,
           'gain_flatmax': 32,
           'gain_min': 0,
           'p_max': 22,
           'pdl': 0,
           'pmd': 0,
           'type_def': 'openroadm_booster',
           'type_variety': 'openroadm_mw_mw_booster'},
          {'allowed_for_design': False,
           'gain_flatmax': 35,
           'gain_min': 25,
           'nf_max': 7,
           'nf_min': 5.5,
           'out_voa_auto': False,
           'p_max': 21,
           'type_def': 'variable_gain',
           'type_variety': 'std_high_gain'},
          {'allowed_for_design': False,
           'gain_flatmax': 26,
           'gain_min': 15,
           'nf_max': 10,
           'nf_min': 6,
           'out_voa_auto': False,
           'p_max': 23,
           'type_def': 'variable_gain',
           'type_variety': 'std_medium_gain'},
          {'allowed_for_design': False,
           'gain_flatmax': 16,
           'gain_min': 8,
           'nf_max': 11,
           'nf_min': 6.5,
           'out_voa_auto': False,
           'p_max': 23,
           'type_def': 'variable_gain',
           'type_variety': 'std_low_gain'},
          {'allowed_for_design': False,
           'gain_flatmax': 16,
           'gain_min': 8,
           'nf_max': 15,
           'nf_min': 9,
           'out_voa_auto': False,
           'p_max': 25,
           'type_def': 'variable_gain',
           'type_variety': 'high_power'},
          {'allowed_for_design': False,
           'gain_flatmax': 21,
           'gain_min': 20,
           'nf0': 5.5,
           'p_max': 21,
           'type_def': 'fixed_gain',
           'type_variety': 'std_fixed_gain'},
          {'allowed_for_design': False,
           'gain_flatmax': 12,
           'gain_min': 12,
           'nf0': -1,
           'p_max': 21,
           'type_def': 'fixed_gain',
           'type_variety': '4pumps_raman'},
          {'allowed_for_design': False,
           'booster_variety': 'std_low_gain',
           'gain_min': 25,
           'preamp_variety': '4pumps_raman',
           'raman': True,
           'type_def': 'dual_stage',
           'type_variety': 'hybrid_4pumps_lowgain'},
          {'allowed_for_design': False,
           'booster_variety': 'std_medium_gain',
           'gain_min': 25,
           'preamp_variety': '4pumps_raman',
           'raman': True,
           'type_def': 'dual_stage',
           'type_variety': 'hybrid_4pumps_mediumgain'},
          {'allowed_for_design': False,
           'booster_variety': 'std_low_gain',
           'gain_min': 25,
           'preamp_variety': 'std_medium_gain',
           'type_def': 'dual_stage',
           'type_variety': 'medium+low_gain'},
          {'allowed_for_design': False,
           'booster_variety': 'high_power',
           'gain_min': 25,
           'preamp_variety': 'std_medium_gain',
           'type_def': 'dual_stage',
           'type_variety': 'medium+high_power'}],
 'Fiber': [{'dispersion': 1.67e-05,
            'effective_area': 8.3e-11,
            'pmd_coef': 1.265e-15,
            'type_variety': 'SSMF'},
           {'dispersion': 5e-06,
            'effective_area': 7.2e-11,
            'pmd_coef': 1.265e-15,
            'type_variety': 'NZDF'},
           {'dispersion': 2.2e-05,
            'effective_area': 1.25e-10,
            'pmd_coef': 1.265e-15,
            'type_variety': 'LOF'},
           {'dispersion': 1.67e-05,
            'effective_area': 8.3e-11,
            'pmd_coef': 1.265e-15,
            'type_variety': 'custom'}],
 'RamanFiber': [{'dispersion': 1.67e-05,
                 'effective_area': 8.3e-11,
                 'pmd_coef': 1.265e-15,
                 'type_variety': 'SSMF'}],
 'Roadm': [{'add_drop_osnr': 33,
            'pdl': 1.5,
            'pmd': 3e-12,
            'restrictions': {'booster_variety_list': ['openroadm_mw_mw_booster'],
                             'preamp_variety_list': ['openroadm_mw_mw_preamp_worstcase_ver5']},
            'target_pch_out_db': -20}],
 'SI': [{'baud_rate': 32000000000.0,
         'f_max': 196100000000000.0,
         'f_min': 191300000000000.0,
         'power_dbm': 0,
         'power_range_db': [0, 0, 1],
         'roll_off': 0.15,
         'spacing': 50000000000.0,
         'sys_margins': 2,
         'tx_osnr': 40}],
 'Span': [{'EOL': 0,
           'con_in': 0,
           'con_out': 0,
           'delta_power_range_db': [-2, 3, 0.5],
           'length_units': 'km',
           'max_fiber_lineic_loss_for_raman': 0.25,
           'max_length': 90,
           'max_loss': 28,
           'padding': 10,
           'power_mode': True,
           'target_extended_gain': 2.5}],
 'Transceiver': [{'frequency': {'max': 196100000000000.0,
                                'min': 191300000000000.0},
                  'mode': [{'OSNR': 12.5,
                            'baud_rate': 30070000000.0,
                            'bit_rate': 100000000000.0,
                            'cost': 1,
                            'format': 'custom',
                            'min_spacing': 50000000000.0,
                            'roll_off': 0.15,
                            'tx_osnr': 34}],
                  'type_variety': 'custom'},
                 {'frequency': {'max': 195100000000000.0,
                                'min': 191300000000000.0},
                  'mode': [{'OSNR': 26,
                            'baud_rate': 59840000000.0,
                            'bit_rate': 400000000000.0,
                            'cost': 1,
                            'format': '400 Gbit/s, DP-16QAM',
                            'min_spacing': 67000000000.0,
                            'roll_off': 0.15,
                            'tx_osnr': 34}],
                  'type_variety': 'OIF 400ZR'},
                 {'frequency': {'max': 196100000000000.0,
                                'min': 191300000000000.0},
                  'mode': [{'OSNR': 12.5,
                            'baud_rate': 30070000000.0,
                            'bit_rate': 100000000000.0,
                            'cost': 1,
                            'format': '100ZR+, DP-QPSK',
                            'min_spacing': 31000000000.0,
                            'roll_off': 0.15,
                            'tx_osnr': 34},
                           {'OSNR': 16,
                            'baud_rate': 60140000000.0,
                            'bit_rate': 200000000000.0,
                            'cost': 1,
                            'format': '200ZR+, DP-QPSK',
                            'min_spacing': 67000000000.0,
                            'roll_off': 0.15,
                            'tx_osnr': 34},
                           {'OSNR': 21,
                            'baud_rate': 60140000000.0,
                            'bit_rate': 300000000000.0,
                            'cost': 1,
                            'format': '300ZR+, DP-8QAM',
                            'min_spacing': 67000000000.0,
                            'roll_off': 0.15,
                            'tx_osnr': 34},
                           {'OSNR': 24,
                            'baud_rate': 60140000000.0,
                            'bit_rate': 400000000000.0,
                            'cost': 1,
                            'format': '400ZR+, DP-16QAM',
                            'min_spacing': 67000000000.0,
                            'roll_off': 0.15,
                            'tx_osnr': 34}],
                  'type_variety': 'OpenZR+'},
                 {'frequency': {'max': 196100000000000.0,
                                'min': 191350000000000.0},
                  'mode': [{'OSNR': 17,
                            'baud_rate': 27950000000.0,
                            'bit_rate': 100000000000.0,
                            'cost': 1,
                            'format': '100 Gbit/s, 27.95 Gbaud, DP-QPSK',
                            'min_spacing': 50000000000.0,
                            'penalties': [{'chromatic_dispersion': 4000.0,
                                           'penalty_value': 0},
                                          {'chromatic_dispersion': 18000.0,
                                           'penalty_value': 0.5},
                                          {'penalty_value': 0, 'pmd': 10},
                                          {'penalty_value': 0.5, 'pmd': 30},
                                          {'pdl': 1, 'penalty_value': 0.5},
                                          {'pdl': 2, 'penalty_value': 1},
                                          {'pdl': 4, 'penalty_value': 2.5},
                                          {'pdl': 6, 'penalty_value': 4}],
                            'roll_off': None,
                            'tx_osnr': 33},
                           {'OSNR': 12,
                            'baud_rate': 31570000000.0,
                            'bit_rate': 100000000000.0,
                            'cost': 1,
                            'format': '100 Gbit/s, 31.57 Gbaud, DP-QPSK',
                            'min_spacing': 50000000000.0,
                            'penalties': [{'chromatic_dispersion': -1000.0,
                                           'penalty_value': 0},
                                          {'chromatic_dispersion': 4000.0,
                                           'penalty_value': 0},
                                          {'chromatic_dispersion': 48000.0,
                                           'penalty_value': 0.5},
                                          {'penalty_value': 0, 'pmd': 10},
                                          {'penalty_value': 0.5, 'pmd': 30},
                                          {'pdl': 1, 'penalty_value': 0.5},
                                          {'pdl': 2, 'penalty_value': 1},
                                          {'pdl': 4, 'penalty_value': 2.5},
                                          {'pdl': 6, 'penalty_value': 4}],
                            'roll_off': 0.15,
                            'tx_osnr': 36},
                           {'OSNR': 20.5,
                            'baud_rate': 31570000000.0,
                            'bit_rate': 100000000000.0,
                            'cost': 1,
                            'format': '200 Gbit/s, 31.57 Gbaud, DP-16QAM',
                            'min_spacing': 50000000000.0,
                            'penalties': [{'chromatic_dispersion': -1000.0,
                                           'penalty_value': 0},
                                          {'chromatic_dispersion': 4000.0,
                                           'penalty_value': 0},
                                          {'chromatic_dispersion': 24000.0,
                                           'penalty_value': 0.5},
                                          {'penalty_value': 0, 'pmd': 10},
                                          {'penalty_value': 0.5, 'pmd': 30},
                                          {'pdl': 1, 'penalty_value': 0.5},
                                          {'pdl': 2, 'penalty_value': 1},
                                          {'pdl': 4, 'penalty_value': 2.5},
                                          {'pdl': 6, 'penalty_value': 4}],
                            'roll_off': 0.15,
                            'tx_osnr': 36},
                           {'OSNR': 17,
                            'baud_rate': 63100000000.0,
                            'bit_rate': 200000000000.0,
                            'cost': 1,
                            'format': '200 Gbit/s, DP-QPSK',
                            'min_spacing': 67000000000.0,
                            'penalties': [{'chromatic_dispersion': -1000.0,
                                           'penalty_value': 0},
                                          {'chromatic_dispersion': 4000.0,
                                           'penalty_value': 0},
                                          {'chromatic_dispersion': 24000.0,
                                           'penalty_value': 0.5},
                                          {'penalty_value': 0, 'pmd': 10},
                                          {'penalty_value': 0.5, 'pmd': 25},
                                          {'pdl': 1, 'penalty_value': 0.5},
                                          {'pdl': 2, 'penalty_value': 1},
                                          {'pdl': 4, 'penalty_value': 2.5}],
                            'roll_off': 0.15,
                            'tx_osnr': 36},
                           {'OSNR': 21,
                            'baud_rate': 63100000000.0,
                            'bit_rate': 300000000000.0,
                            'cost': 1,
                            'format': '300 Gbit/s, DP-8QAM',
                            'min_spacing': 50000000000.0,
                            'penalties': [{'chromatic_dispersion': -1000.0,
                                           'penalty_value': 0},
                                          {'chromatic_dispersion': 4000.0,
                                           'penalty_value': 0},
                                          {'chromatic_dispersion': 18000.0,
                                           'penalty_value': 0.5},
                                          {'penalty_value': 0, 'pmd': 10},
                                          {'penalty_value': 0.5, 'pmd': 25},
                                          {'pdl': 1, 'penalty_value': 0.5},
                                          {'pdl': 2, 'penalty_value': 1},
                                          {'pdl': 4, 'penalty_value': 2.5}],
                            'roll_off': 0.15,
                            'tx_osnr': 36},
                           {'OSNR': 24,
                            'baud_rate': 63100000000.0,
                            'bit_rate': 400000000000.0,
                            'cost': 1,
                            'format': '400 Gbit/s, DP-16QAM',
                            'min_spacing': 67000000000.0,
                            'penalties': [{'chromatic_dispersion': -1000.0,
                                           'penalty_value': 0},
                                          {'chromatic_dispersion': 4000.0,
                                           'penalty_value': 0},
                                          {'chromatic_dispersion': 12000.0,
                                           'penalty_value': 0.5},
                                          {'penalty_value': 0, 'pmd': 10},
                                          {'penalty_value': 0.5, 'pmd': 20},
                                          {'pdl': 1, 'penalty_value': 0.5},
                                          {'pdl': 2, 'penalty_value': 1},
                                          {'pdl': 4, 'penalty_value': 2.5}],
                            'roll_off': 0.15,
                            'tx_osnr': 36}],
                  'type_variety': 'OpenROADM MSA ver. 5.0'},
                 {'frequency': {'max': 196100000000000.0,
                                'min': 191350000000000.0},
                  'mode': [{'OSNR': 17,
                            'baud_rate': 27950000000.0,
                            'bit_rate': 100000000000.0,
                            'cost': 1,
                            'format': '100 Gbit/s, 27.95 Gbaud, DP-QPSK',
                            'min_spacing': 50000000000.0,
                            'penalties': [{'chromatic_dispersion': 4000.0,
                                           'penalty_value': 0},
                                          {'chromatic_dispersion': 18000.0,
                                           'penalty_value': 0.5},
                                          {'penalty_value': 0, 'pmd': 10},
                                          {'penalty_value': 0.5, 'pmd': 30},
                                          {'pdl': 1, 'penalty_value': 0.5},
                                          {'pdl': 2, 'penalty_value': 1},
                                          {'pdl': 4, 'penalty_value': 2.5},
                                          {'pdl': 6, 'penalty_value': 4}],
                            'roll_off': None,
                            'tx_osnr': 33},
                           {'OSNR': 12,
                            'baud_rate': 31570000000.0,
                            'bit_rate': 100000000000.0,
                            'cost': 1,
                            'format': '100 Gbit/s, 31.57 Gbaud, DP-QPSK',
                            'min_spacing': 50000000000.0,
                            'penalties': [{'chromatic_dispersion': -1000.0,
                                           'penalty_value': 0},
                                          {'chromatic_dispersion': 4000.0,
                                           'penalty_value': 0},
                                          {'chromatic_dispersion': 40000.0,
                                           'penalty_value': 0.5},
                                          {'penalty_value': 0, 'pmd': 10},
                                          {'penalty_value': 0.5, 'pmd': 30},
                                          {'pdl': 1, 'penalty_value': 0.5},
                                          {'pdl': 2, 'penalty_value': 1},
                                          {'pdl': 4, 'penalty_value': 2.5},
                                          {'pdl': 6, 'penalty_value': 4}],
                            'roll_off': 0.15,
                            'tx_osnr': 35},
                           {'OSNR': 17,
                            'baud_rate': 63100000000.0,
                            'bit_rate': 200000000000.0,
                            'cost': 1,
                            'format': '200 Gbit/s, DP-QPSK',
                            'min_spacing': 67000000000.0,
                            'penalties': [{'chromatic_dispersion': -1000.0,
                                           'penalty_value': 0},
                                          {'chromatic_dispersion': 4000.0,
                                           'penalty_value': 0},
                                          {'chromatic_dispersion': 24000.0,
                                           'penalty_value': 0.5},
                                          {'penalty_value': 0, 'pmd': 10},
                                          {'penalty_value': 0.5, 'pmd': 25},
                                          {'pdl': 1, 'penalty_value': 0.5},
                                          {'pdl': 2, 'penalty_value': 1},
                                          {'pdl': 4, 'penalty_value': 2.5}],
                            'roll_off': 0.15,
                            'tx_osnr': 36},
                           {'OSNR': 21,
                            'baud_rate': 63100000000.0,
                            'bit_rate': 300000000000.0,
                            'cost': 1,
                            'format': '300 Gbit/s, DP-8QAM',
                            'min_spacing': 67000000000.0,
                            'penalties': [{'chromatic_dispersion': -1000.0,
                                           'penalty_value': 0},
                                          {'chromatic_dispersion': 4000.0,
                                           'penalty_value': 0},
                                          {'chromatic_dispersion': 18000.0,
                                           'penalty_value': 0.5},
                                          {'penalty_value': 0, 'pmd': 10},
                                          {'penalty_value': 0.5, 'pmd': 25},
                                          {'pdl': 1, 'penalty_value': 0.5},
                                          {'pdl': 2, 'penalty_value': 1},
                                          {'pdl': 4, 'penalty_value': 2.5}],
                            'roll_off': 0.15,
                            'tx_osnr': 36},
                           {'OSNR': 24,
                            'baud_rate': 63100000000.0,
                            'bit_rate': 400000000000.0,
                            'cost': 1,
                            'format': '400 Gbit/s, DP-16QAM',
                            'min_spacing': 67000000000.0,
                            'penalties': [{'chromatic_dispersion': -1000.0,
                                           'penalty_value': 0},
                                          {'chromatic_dispersion': 4000.0,
                                           'penalty_value': 0},
                                          {'chromatic_dispersion': 12000.0,
                                           'penalty_value': 0.5},
                                          {'penalty_value': 0, 'pmd': 10},
                                          {'penalty_value': 0.5, 'pmd': 20},
                                          {'pdl': 1, 'penalty_value': 0.5},
                                          {'pdl': 2, 'penalty_value': 1},
                                          {'pdl': 4, 'penalty_value': 2.5}],
                            'roll_off': 0.15,
                            'tx_osnr': 36}],
                  'type_variety': 'OpenROADM MSA ver. 4.0'},
                 {'frequency': {'max': 195100000000000.0,
                                'min': 191300000000000.0},
                  'mode': [{'OSNR': 14.5,
                            'baud_rate': 27950000000.0,
                            'bit_rate': 100000000000.0,
                            'cost': 1,
                            'format': 'PHYv1.0, 100 Gbit/s, DP-QPSK',
                            'min_spacing': 50000000000.0,
                            'roll_off': 0.15,
                            'tx_osnr': 35},
                           {'OSNR': 15.5,
                            'baud_rate': 63140000000.0,
                            'bit_rate': 200000000000.0,
                            'cost': 1,
                            'format': 'PHYv2.0, 200 Gbit/s, DP-QPSK',
                            'min_spacing': 67000000000.0,
                            'roll_off': 0.15,
                            'tx_osnr': 35}],
                  'type_variety': 'CableLabs P2PCO'},
                 {'frequency': {'max': 195100000000000.0,
                                'min': 191300000000000.0},
                  'mode': [{'OSNR': 19.5,
                            'baud_rate': 27950000000.0,
                            'bit_rate': 100000000000.0,
                            'cost': 1,
                            'format': '100GBASE-ZR',
                            'min_spacing': 50000000000.0,
                            'roll_off': 0.15,
                            'tx_osnr': 35}],
                  'type_variety': 'IEEE'},
                 {'frequency': {'max': 195100000000000.0,
                                'min': 191300000000000.0},
                  'mode': [{'OSNR': 10.5,
                            'baud_rate': 34160000000.0,
                            'bit_rate': 100000000000.0,
                            'cost': 1,
                            'format': '100 Gbit/s, DP-QPSK',
                            'min_spacing': 50000000000.0,
                            'roll_off': 0.15,
                            'tx_osnr': 40},
                           {'OSNR': 15.5,
                            'baud_rate': 34160000000.0,
                            'bit_rate': 150000000000.0,
                            'cost': 1,
                            'format': '150 Gbit/s, DP-8QAM',
                            'min_spacing': 50000000000.0,
                            'roll_off': 0.15,
                            'tx_osnr': 40},
                           {'OSNR': 18.5,
                            'baud_rate': 34160000000.0,
                            'bit_rate': 200000000000.0,
                            'cost': 1,
                            'format': '200 Gbit/s, DP-16QAM',
                            'min_spacing': 50000000000.0,
                            'roll_off': 0.15,
                            'tx_osnr': 40}],
                  'type_variety': 'Voyager'},
                 {'frequency': {'max': 195100000000000.0,
                                'min': 191300000000000.0},
                  'mode': [{'OSNR': 12.3,
                            'baud_rate': 31380000000.0,
                            'bit_rate': 100000000000.0,
                            'cost': 1,
                            'format': '100 Gbit/s, DP-QPSK',
                            'min_spacing': 50000000000.0,
                            'roll_off': 0.15,
                            'tx_osnr': 37},
                           {'OSNR': 20,
                            'baud_rate': 41840000000.0,
                            'bit_rate': 200000000000.0,
                            'cost': 1,
                            'format': '200 Gbit/s, DP-8QAM',
                            'min_spacing': 50000000000.0,
                            'roll_off': 0.15,
                            'tx_osnr': 32.3},
                           {'OSNR': 21.5,
                            'baud_rate': 31380000000.0,
                            'bit_rate': 200000000000.0,
                            'cost': 1,
                            'format': '200 Gbit/s, DP-16QAM',
                            'min_spacing': 50000000000.0,
                            'roll_off': 0.15,
                            'tx_osnr': 34.3}],
                  'type_variety': 'Cisco CFP2-DCO'},
                 {'frequency': {'max': 195100000000000.0,
                                'min': 191300000000000.0},
                  'mode': [{'OSNR': 11,
                            'baud_rate': 69440000000.0,
                            'bit_rate': 100000000000.0,
                            'cost': 1,
                            'format': '100 Gbit/s, DP-BPSK',
                            'min_spacing': 70000000000.0,
                            'roll_off': 0.15,
                            'tx_osnr': 40},
                           {'OSNR': 14.1,
                            'baud_rate': 69440000000.0,
                            'bit_rate': 200000000000.0,
                            'cost': 1,
                            'format': '200 Gbit/s, DP-QPSK',
                            'min_spacing': 70000000000.0,
                            'roll_off': 0.15,
                            'tx_osnr': 40},
                           {'OSNR': 18.5,
                            'baud_rate': 69440000000.0,
                            'bit_rate': 300000000000.0,
                            'cost': 1,
                            'format': '300 Gbit/s, DP-8QAM',
                            'min_spacing': 70000000000.0,
                            'roll_off': 0.15,
                            'tx_osnr': 40},
                           {'OSNR': 22,
                            'baud_rate': 69440000000.0,
                            'bit_rate': 400000000000.0,
                            'cost': 1,
                            'format': '400 Gbit/s, DP-16QAM',
                            'min_spacing': 70000000000.0,
                            'roll_off': 0.15,
                            'tx_osnr': 40},
                           {'OSNR': 27.3,
                            'baud_rate': 69440000000.0,
                            'bit_rate': 500000000000.0,
                            'cost': 1,
                            'format': '500 Gbit/s, DP-32QAM',
                            'min_spacing': 70000000000.0,
                            'roll_off': 0.15,
                            'tx_osnr': 40},
                           {'OSNR': 32.5,
                            'baud_rate': 71850000000.0,
                            'bit_rate': 600000000000.0,
                            'cost': 1,
                            'format': '600 Gbit/s, DP-32QAM-64QAM hybrid',
                            'min_spacing': 75000000000.0,
                            'roll_off': 0.15,
                            'tx_osnr': 40}],
                  'type_variety': 'Cisco NCS 1004'},
                 {'frequency': {'max': 195100000000000.0,
                                'min': 191300000000000.0},
                  'mode': [{'OSNR': 10.5,
                            'baud_rate': 34170000000.0,
                            'bit_rate': 100000000000.0,
                            'cost': 1,
                            'format': '100 Gbit/s, DP-QPSK',
                            'min_spacing': 50000000000.0,
                            'roll_off': 0.15,
                            'tx_osnr': 36},
                           {'OSNR': 15.5,
                            'baud_rate': 34170000000.0,
                            'bit_rate': 150000000000.0,
                            'cost': 1,
                            'format': '150 Gbit/s, DP-8QAM',
                            'min_spacing': 50000000000.0,
                            'roll_off': 0.15,
                            'tx_osnr': 36},
                           {'OSNR': 18.5,
                            'baud_rate': 34170000000.0,
                            'bit_rate': 200000000000.0,
                            'cost': 1,
                            'format': '200 Gbit/s, DP-16QAM',
                            'min_spacing': 50000000000.0,
                            'roll_off': 0.15,
                            'tx_osnr': 36}],
                  'type_variety': 'Juniper QFX10000'}]}
    eqpt_path_02 = 'eqpt_config_IKR_wrap.json'
    path_to_equipment = eqpt_path_02
    filename= str(0) #'test-by-jensk' #Path(path_to_equipment)
    equipment_02= _equipment_from_json(json_data_03, filename)
    
    path_request_02 = create_path_request(equipment_02, req_params_mesh)
    path,infos, infos_02, si_list_02 = simulate(equipment_02, network, path_request_02)
    print_path(path)
    x = final_gsnr(path)
    print('Final GSNR in dB')
    print(x)
    
elif number_of_simulation == 1002:
    topology_path_02 = 'LinearTopology.json'
    path_to_network_02= topology_path_02
    filename_02_nw = Path(path_to_network_02)
    nw_json_data= load_json(filename_02_nw)
    print('Type of nw json data:')
    print(type(nw_json_data))
    pprint.pprint(nw_json_data)
    new_nw_definition= {'connections': [{'from_node': 'roadm Lannion_CAS',
                  'to_node': 'east edfa in Lannion_CAS to Corlay'},
                 {'from_node': 'east edfa in Lannion_CAS to Corlay',
                  'to_node': 'fiber (Lannion_CAS → Corlay)-F061'},
                 {'from_node': 'fiber (Corlay → Lannion_CAS)-F061',
                  'to_node': 'west edfa in Lannion_CAS to Corlay'},
                 {'from_node': 'west edfa in Lannion_CAS to Corlay',
                  'to_node': 'roadm Lannion_CAS'},
                 {'from_node': 'roadm Lannion_CAS',
                  'to_node': 'east edfa in Lannion_CAS to Stbrieuc'},
                 {'from_node': 'east edfa in Lannion_CAS to Stbrieuc',
                  'to_node': 'fiber (Lannion_CAS → Stbrieuc)-F056'},
                 {'from_node': 'fiber (Stbrieuc → Lannion_CAS)-F056',
                  'to_node': 'west edfa in Lannion_CAS to Stbrieuc'},
                 {'from_node': 'west edfa in Lannion_CAS to Stbrieuc',
                  'to_node': 'roadm Lannion_CAS'},
                 {'from_node': 'roadm Lannion_CAS',
                  'to_node': 'east edfa in Lannion_CAS to Morlaix'},
                 {'from_node': 'east edfa in Lannion_CAS to Morlaix',
                  'to_node': 'fiber (Lannion_CAS → Morlaix)-F059'},
                 {'from_node': 'fiber (Morlaix → Lannion_CAS)-F059',
                  'to_node': 'west edfa in Lannion_CAS to Morlaix'},
                 {'from_node': 'west edfa in Lannion_CAS to Morlaix',
                  'to_node': 'roadm Lannion_CAS'},
                 {'from_node': 'fiber (Lannion_CAS → Corlay)-F061',
                  'to_node': 'west fused spans in Corlay'},
                 {'from_node': 'west fused spans in Corlay',
                  'to_node': 'fiber (Corlay → Loudeac)-F010'},
                 {'from_node': 'fiber (Loudeac → Corlay)-F010',
                  'to_node': 'east fused spans in Corlay'},
                 {'from_node': 'east fused spans in Corlay',
                  'to_node': 'fiber (Corlay → Lannion_CAS)-F061'},
                 {'from_node': 'fiber (Corlay → Loudeac)-F010',
                  'to_node': 'west fused spans in Loudeac'},
                 {'from_node': 'west fused spans in Loudeac',
                  'to_node': 'fiber (Loudeac → Lorient_KMA)-F054'},
                 {'from_node': 'fiber (Lorient_KMA → Loudeac)-F054',
                  'to_node': 'east fused spans in Loudeac'},
                 {'from_node': 'east fused spans in Loudeac',
                  'to_node': 'fiber (Loudeac → Corlay)-F010'},
                 {'from_node': 'roadm Lorient_KMA',
                  'to_node': 'east edfa in Lorient_KMA to Loudeac'},
                 {'from_node': 'east edfa in Lorient_KMA to Loudeac',
                  'to_node': 'fiber (Lorient_KMA → Loudeac)-F054'},
                 {'from_node': 'fiber (Loudeac → Lorient_KMA)-F054',
                  'to_node': 'west edfa in Lorient_KMA to Loudeac'},
                 {'from_node': 'west edfa in Lorient_KMA to Loudeac',
                  'to_node': 'roadm Lorient_KMA'},
                 {'from_node': 'roadm Lorient_KMA',
                  'to_node': 'east edfa in Lorient_KMA to Vannes_KBE'},
                 {'from_node': 'east edfa in Lorient_KMA to Vannes_KBE',
                  'to_node': 'fiber (Lorient_KMA → Vannes_KBE)-F055'},
                 {'from_node': 'fiber (Vannes_KBE → Lorient_KMA)-F055',
                  'to_node': 'west edfa in Lorient_KMA to Vannes_KBE'},
                 {'from_node': 'west edfa in Lorient_KMA to Vannes_KBE',
                  'to_node': 'roadm Lorient_KMA'},
                 {'from_node': 'roadm Lorient_KMA',
                  'to_node': 'fiber (Lorient_KMA → Quimper)-'},
                 {'from_node': 'fiber (Quimper → Lorient_KMA)-',
                  'to_node': 'roadm Lorient_KMA'},
                 {'from_node': 'roadm Vannes_KBE',
                  'to_node': 'east edfa in Vannes_KBE to Lorient_KMA'},
                 {'from_node': 'east edfa in Vannes_KBE to Lorient_KMA',
                  'to_node': 'fiber (Vannes_KBE → Lorient_KMA)-F055'},
                 {'from_node': 'fiber (Lorient_KMA → Vannes_KBE)-F055',
                  'to_node': 'west edfa in Vannes_KBE to Lorient_KMA'},
                 {'from_node': 'west edfa in Vannes_KBE to Lorient_KMA',
                  'to_node': 'roadm Vannes_KBE'},
                 {'from_node': 'roadm Vannes_KBE',
                  'to_node': 'fiber (Vannes_KBE → Ploermel)-'},
                 {'from_node': 'fiber (Ploermel → Vannes_KBE)-',
                  'to_node': 'roadm Vannes_KBE'},
                 {'from_node': 'fiber (Lannion_CAS → Stbrieuc)-F056',
                  'to_node': 'east edfa in Stbrieuc to Rennes_STA'},
                 {'from_node': 'east edfa in Stbrieuc to Rennes_STA',
                  'to_node': 'fiber (Stbrieuc → Rennes_STA)-F057'},
                 {'from_node': 'fiber (Rennes_STA → Stbrieuc)-F057',
                  'to_node': 'west edfa in Stbrieuc to Rennes_STA'},
                 {'from_node': 'west edfa in Stbrieuc to Rennes_STA',
                  'to_node': 'fiber (Stbrieuc → Lannion_CAS)-F056'},
                 {'from_node': 'roadm Rennes_STA',
                  'to_node': 'east edfa in Rennes_STA to Stbrieuc'},
                 {'from_node': 'east edfa in Rennes_STA to Stbrieuc',
                  'to_node': 'fiber (Rennes_STA → Stbrieuc)-F057'},
                 {'from_node': 'fiber (Stbrieuc → Rennes_STA)-F057',
                  'to_node': 'west edfa in Rennes_STA to Stbrieuc'},
                 {'from_node': 'west edfa in Rennes_STA to Stbrieuc',
                  'to_node': 'roadm Rennes_STA'},
                 {'from_node': 'roadm Rennes_STA',
                  'to_node': 'fiber (Rennes_STA → Ploermel)-'},
                 {'from_node': 'fiber (Ploermel → Rennes_STA)-',
                  'to_node': 'roadm Rennes_STA'},
                 {'from_node': 'fiber (Lannion_CAS → Morlaix)-F059',
                  'to_node': 'west fused spans in Morlaix'},
                 {'from_node': 'west fused spans in Morlaix',
                  'to_node': 'fiber (Morlaix → Brest_KLA)-F060'},
                 {'from_node': 'fiber (Brest_KLA → Morlaix)-F060',
                  'to_node': 'east fused spans in Morlaix'},
                 {'from_node': 'east fused spans in Morlaix',
                  'to_node': 'fiber (Morlaix → Lannion_CAS)-F059'},
                 {'from_node': 'roadm Brest_KLA',
                  'to_node': 'east edfa in Brest_KLA to Morlaix'},
                 {'from_node': 'east edfa in Brest_KLA to Morlaix',
                  'to_node': 'fiber (Brest_KLA → Morlaix)-F060'},
                 {'from_node': 'fiber (Morlaix → Brest_KLA)-F060',
                  'to_node': 'west edfa in Brest_KLA to Morlaix'},
                 {'from_node': 'west edfa in Brest_KLA to Morlaix',
                  'to_node': 'roadm Brest_KLA'},
                 {'from_node': 'roadm Brest_KLA',
                  'to_node': 'fiber (Brest_KLA → Quimper)-'},
                 {'from_node': 'fiber (Quimper → Brest_KLA)-',
                  'to_node': 'roadm Brest_KLA'},
                 {'from_node': 'fiber (Brest_KLA → Quimper)-',
                  'to_node': 'west edfa in Quimper'},
                 {'from_node': 'west edfa in Quimper',
                  'to_node': 'fiber (Quimper → Lorient_KMA)-'},
                 {'from_node': 'fiber (Lorient_KMA → Quimper)-',
                  'to_node': 'east edfa in Quimper'},
                 {'from_node': 'east edfa in Quimper',
                  'to_node': 'fiber (Quimper → Brest_KLA)-'},
                 {'from_node': 'fiber (Vannes_KBE → Ploermel)-',
                  'to_node': 'west edfa in Ploermel'},
                 {'from_node': 'west edfa in Ploermel',
                  'to_node': 'fiber (Ploermel → Rennes_STA)-'},
                 {'from_node': 'fiber (Rennes_STA → Ploermel)-',
                  'to_node': 'east edfa in Ploermel'},
                 {'from_node': 'east edfa in Ploermel',
                  'to_node': 'fiber (Ploermel → Vannes_KBE)-'},
                 {'from_node': 'trx Lannion_CAS',
                  'to_node': 'roadm Lannion_CAS'},
                 {'from_node': 'roadm Lannion_CAS',
                  'to_node': 'trx Lannion_CAS'},
                 {'from_node': 'trx Lorient_KMA',
                  'to_node': 'roadm Lorient_KMA'},
                 {'from_node': 'roadm Lorient_KMA',
                  'to_node': 'trx Lorient_KMA'},
                 {'from_node': 'trx Vannes_KBE', 'to_node': 'roadm Vannes_KBE'},
                 {'from_node': 'roadm Vannes_KBE', 'to_node': 'trx Vannes_KBE'},
                 {'from_node': 'trx Rennes_STA', 'to_node': 'roadm Rennes_STA'},
                 {'from_node': 'roadm Rennes_STA', 'to_node': 'trx Rennes_STA'},
                 {'from_node': 'trx Brest_KLA', 'to_node': 'roadm Brest_KLA'},
                 {'from_node': 'roadm Brest_KLA', 'to_node': 'trx Brest_KLA'}],
 'elements': [{'metadata': {'location': {'city': 'Lannion_CAS',
                                         'latitude': 2.0,
                                         'longitude': 0.0,
                                         'region': 'RLD'}},
               'type': 'Transceiver',
               'uid': 'trx Lannion_CAS'},
              {'metadata': {'location': {'city': 'Lorient_KMA',
                                         'latitude': 2.0,
                                         'longitude': 3.0,
                                         'region': 'RLD'}},
               'type': 'Transceiver',
               'uid': 'trx Lorient_KMA'},
              {'metadata': {'location': {'city': 'Vannes_KBE',
                                         'latitude': 2.0,
                                         'longitude': 4.0,
                                         'region': 'RLD'}},
               'type': 'Transceiver',
               'uid': 'trx Vannes_KBE'},
              {'metadata': {'location': {'city': 'Rennes_STA',
                                         'latitude': 0.0,
                                         'longitude': 0.0,
                                         'region': 'RLD'}},
               'type': 'Transceiver',
               'uid': 'trx Rennes_STA'},
              {'metadata': {'location': {'city': 'Brest_KLA',
                                         'latitude': 4.0,
                                         'longitude': 0.0,
                                         'region': 'RLD'}},
               'type': 'Transceiver',
               'uid': 'trx Brest_KLA'},
              {'metadata': {'location': {'city': 'Lannion_CAS',
                                         'latitude': 2.0,
                                         'longitude': 0.0,
                                         'region': 'RLD'}},
               'type': 'Roadm',
               'uid': 'roadm Lannion_CAS'},
              {'metadata': {'location': {'city': 'Lorient_KMA',
                                         'latitude': 2.0,
                                         'longitude': 3.0,
                                         'region': 'RLD'}},
               'type': 'Roadm',
               'uid': 'roadm Lorient_KMA'},
              {'metadata': {'location': {'city': 'Vannes_KBE',
                                         'latitude': 2.0,
                                         'longitude': 4.0,
                                         'region': 'RLD'}},
               'type': 'Roadm',
               'uid': 'roadm Vannes_KBE'},
              {'metadata': {'location': {'city': 'Rennes_STA',
                                         'latitude': 0.0,
                                         'longitude': 0.0,
                                         'region': 'RLD'}},
               'type': 'Roadm',
               'uid': 'roadm Rennes_STA'},
              {'metadata': {'location': {'city': 'Brest_KLA',
                                         'latitude': 4.0,
                                         'longitude': 0.0,
                                         'region': 'RLD'}},
               'type': 'Roadm',
               'uid': 'roadm Brest_KLA'},
              {'metadata': {'location': {'city': 'Corlay',
                                         'latitude': 2.0,
                                         'longitude': 1.0,
                                         'region': 'RLD'}},
               'type': 'Fused',
               'uid': 'west fused spans in Corlay'},
              {'metadata': {'location': {'city': 'Loudeac',
                                         'latitude': 2.0,
                                         'longitude': 2.0,
                                         'region': 'RLD'}},
               'type': 'Fused',
               'uid': 'west fused spans in Loudeac'},
              {'metadata': {'location': {'city': 'Morlaix',
                                         'latitude': 3.0,
                                         'longitude': 0.0,
                                         'region': 'RLD'}},
               'type': 'Fused',
               'uid': 'west fused spans in Morlaix'},
              {'metadata': {'location': {'city': 'Corlay',
                                         'latitude': 2.0,
                                         'longitude': 1.0,
                                         'region': 'RLD'}},
               'type': 'Fused',
               'uid': 'east fused spans in Corlay'},
              {'metadata': {'location': {'city': 'Loudeac',
                                         'latitude': 2.0,
                                         'longitude': 2.0,
                                         'region': 'RLD'}},
               'type': 'Fused',
               'uid': 'east fused spans in Loudeac'},
              {'metadata': {'location': {'city': 'Morlaix',
                                         'latitude': 3.0,
                                         'longitude': 0.0,
                                         'region': 'RLD'}},
               'type': 'Fused',
               'uid': 'east fused spans in Morlaix'},
              {'metadata': {'location': {'latitude': 2.0, 'longitude': 0.5}},
               'params': {'con_in': None,
                          'con_out': None,
                          'length': 20.0,
                          'length_units': 'km',
                          'loss_coef': 0.2},
               'type': 'Fiber',
               'type_variety': 'SSMF',
               'uid': 'fiber (Lannion_CAS → Corlay)-F061'},
              {'metadata': {'location': {'latitude': 2.0, 'longitude': 1.5}},
               'params': {'con_in': None,
                          'con_out': None,
                          'length': 50.0,
                          'length_units': 'km',
                          'loss_coef': 0.2},
               'type': 'Fiber',
               'type_variety': 'SSMF',
               'uid': 'fiber (Corlay → Loudeac)-F010'},
              {'metadata': {'location': {'latitude': 2.0, 'longitude': 2.5}},
               'params': {'con_in': None,
                          'con_out': None,
                          'length': 60.0,
                          'length_units': 'km',
                          'loss_coef': 0.2},
               'type': 'Fiber',
               'type_variety': 'SSMF',
               'uid': 'fiber (Loudeac → Lorient_KMA)-F054'},
              {'metadata': {'location': {'latitude': 2.0, 'longitude': 3.5}},
               'params': {'con_in': None,
                          'con_out': None,
                          'length': 10.0,
                          'length_units': 'km',
                          'loss_coef': 0.2},
               'type': 'Fiber',
               'type_variety': 'SSMF',
               'uid': 'fiber (Lorient_KMA → Vannes_KBE)-F055'},
              {'metadata': {'location': {'latitude': 1.5, 'longitude': 0.0}},
               'params': {'con_in': None,
                          'con_out': None,
                          'length': 60.0,
                          'length_units': 'km',
                          'loss_coef': 0.2},
               'type': 'Fiber',
               'type_variety': 'SSMF',
               'uid': 'fiber (Lannion_CAS → Stbrieuc)-F056'},
              {'metadata': {'location': {'latitude': 0.5, 'longitude': 0.0}},
               'params': {'con_in': None,
                          'con_out': None,
                          'length': 65.0,
                          'length_units': 'km',
                          'loss_coef': 0.2},
               'type': 'Fiber',
               'type_variety': 'SSMF',
               'uid': 'fiber (Stbrieuc → Rennes_STA)-F057'},
              {'metadata': {'location': {'latitude': 2.5, 'longitude': 0.0}},
               'params': {'con_in': None,
                          'con_out': None,
                          'length': 40.0,
                          'length_units': 'km',
                          'loss_coef': 0.2},
               'type': 'Fiber',
               'type_variety': 'SSMF',
               'uid': 'fiber (Lannion_CAS → Morlaix)-F059'},
              {'metadata': {'location': {'latitude': 3.5, 'longitude': 0.0}},
               'params': {'con_in': None,
                          'con_out': None,
                          'length': 35.0,
                          'length_units': 'km',
                          'loss_coef': 0.2},
               'type': 'Fiber',
               'type_variety': 'SSMF',
               'uid': 'fiber (Morlaix → Brest_KLA)-F060'},
              {'metadata': {'location': {'latitude': 2.5, 'longitude': 0.5}},
               'params': {'con_in': None,
                          'con_out': None,
                          'length': 75.0,
                          'length_units': 'km',
                          'loss_coef': 0.2},
               'type': 'Fiber',
               'type_variety': 'SSMF',
               'uid': 'fiber (Brest_KLA → Quimper)-'},
              {'metadata': {'location': {'latitude': 1.5, 'longitude': 2.0}},
               'params': {'con_in': None,
                          'con_out': None,
                          'length': 70.0,
                          'length_units': 'km',
                          'loss_coef': 0.2},
               'type': 'Fiber',
               'type_variety': 'SSMF',
               'uid': 'fiber (Quimper → Lorient_KMA)-'},
              {'metadata': {'location': {'latitude': 1.5, 'longitude': 3.0}},
               'params': {'con_in': None,
                          'con_out': None,
                          'length': 50.0,
                          'length_units': 'km',
                          'loss_coef': 0.2},
               'type': 'Fiber',
               'type_variety': 'SSMF',
               'uid': 'fiber (Ploermel → Vannes_KBE)-'},
              {'metadata': {'location': {'latitude': 0.5, 'longitude': 1.0}},
               'params': {'con_in': None,
                          'con_out': None,
                          'length': 55.0,
                          'length_units': 'km',
                          'loss_coef': 0.2},
               'type': 'Fiber',
               'type_variety': 'SSMF',
               'uid': 'fiber (Ploermel → Rennes_STA)-'},
              {'metadata': {'location': {'latitude': 2.0, 'longitude': 0.5}},
               'params': {'con_in': None,
                          'con_out': None,
                          'length': 20.0,
                          'length_units': 'km',
                          'loss_coef': 0.2},
               'type': 'Fiber',
               'type_variety': 'SSMF',
               'uid': 'fiber (Corlay → Lannion_CAS)-F061'},
              {'metadata': {'location': {'latitude': 2.0, 'longitude': 1.5}},
               'params': {'con_in': None,
                          'con_out': None,
                          'length': 50.0,
                          'length_units': 'km',
                          'loss_coef': 0.2},
               'type': 'Fiber',
               'type_variety': 'SSMF',
               'uid': 'fiber (Loudeac → Corlay)-F010'},
              {'metadata': {'location': {'latitude': 2.0, 'longitude': 2.5}},
               'params': {'con_in': None,
                          'con_out': None,
                          'length': 60.0,
                          'length_units': 'km',
                          'loss_coef': 0.2},
               'type': 'Fiber',
               'type_variety': 'SSMF',
               'uid': 'fiber (Lorient_KMA → Loudeac)-F054'},
              {'metadata': {'location': {'latitude': 2.0, 'longitude': 3.5}},
               'params': {'con_in': None,
                          'con_out': None,
                          'length': 10.0,
                          'length_units': 'km',
                          'loss_coef': 0.2},
               'type': 'Fiber',
               'type_variety': 'SSMF',
               'uid': 'fiber (Vannes_KBE → Lorient_KMA)-F055'},
              {'metadata': {'location': {'latitude': 1.5, 'longitude': 0.0}},
               'params': {'con_in': None,
                          'con_out': None,
                          'length': 60.0,
                          'length_units': 'km',
                          'loss_coef': 0.2},
               'type': 'Fiber',
               'type_variety': 'SSMF',
               'uid': 'fiber (Stbrieuc → Lannion_CAS)-F056'},
              {'metadata': {'location': {'latitude': 0.5, 'longitude': 0.0}},
               'params': {'con_in': None,
                          'con_out': None,
                          'length': 65.0,
                          'length_units': 'km',
                          'loss_coef': 0.2},
               'type': 'Fiber',
               'type_variety': 'SSMF',
               'uid': 'fiber (Rennes_STA → Stbrieuc)-F057'},
              {'metadata': {'location': {'latitude': 2.5, 'longitude': 0.0}},
               'params': {'con_in': None,
                          'con_out': None,
                          'length': 40.0,
                          'length_units': 'km',
                          'loss_coef': 0.2},
               'type': 'Fiber',
               'type_variety': 'SSMF',
               'uid': 'fiber (Morlaix → Lannion_CAS)-F059'},
              {'metadata': {'location': {'latitude': 3.5, 'longitude': 0.0}},
               'params': {'con_in': None,
                          'con_out': None,
                          'length': 35.0,
                          'length_units': 'km',
                          'loss_coef': 0.2},
               'type': 'Fiber',
               'type_variety': 'SSMF',
               'uid': 'fiber (Brest_KLA → Morlaix)-F060'},
              {'metadata': {'location': {'latitude': 2.5, 'longitude': 0.5}},
               'params': {'con_in': None,
                          'con_out': None,
                          'length': 75.0,
                          'length_units': 'km',
                          'loss_coef': 0.2},
               'type': 'Fiber',
               'type_variety': 'SSMF',
               'uid': 'fiber (Quimper → Brest_KLA)-'},
              {'metadata': {'location': {'latitude': 1.5, 'longitude': 2.0}},
               'params': {'con_in': None,
                          'con_out': None,
                          'length': 70.0,
                          'length_units': 'km',
                          'loss_coef': 0.2},
               'type': 'Fiber',
               'type_variety': 'SSMF',
               'uid': 'fiber (Lorient_KMA → Quimper)-'},
              {'metadata': {'location': {'latitude': 1.5, 'longitude': 3.0}},
               'params': {'con_in': None,
                          'con_out': None,
                          'length': 50.0,
                          'length_units': 'km',
                          'loss_coef': 0.2},
               'type': 'Fiber',
               'type_variety': 'SSMF',
               'uid': 'fiber (Vannes_KBE → Ploermel)-'},
              {'metadata': {'location': {'latitude': 0.5, 'longitude': 1.0}},
               'params': {'con_in': None,
                          'con_out': None,
                          'length': 55.0,
                          'length_units': 'km',
                          'loss_coef': 0.2},
               'type': 'Fiber',
               'type_variety': 'SSMF',
               'uid': 'fiber (Rennes_STA → Ploermel)-'},
              {'metadata': {'location': {'city': 'Quimper',
                                         'latitude': 1.0,
                                         'longitude': 1.0,
                                         'region': 'RLD'}},
               'operational': {'gain_target': None, 'tilt_target': 0},
               'type': 'Edfa',
               'uid': 'west edfa in Quimper'},
              {'metadata': {'location': {'city': 'Ploermel',
                                         'latitude': 1.0,
                                         'longitude': 2.0,
                                         'region': 'RLD'}},
               'operational': {'gain_target': None, 'tilt_target': 0},
               'type': 'Edfa',
               'uid': 'west edfa in Ploermel'},
              {'metadata': {'location': {'city': 'Quimper',
                                         'latitude': 1.0,
                                         'longitude': 1.0,
                                         'region': 'RLD'}},
               'operational': {'gain_target': None, 'tilt_target': 0},
               'type': 'Edfa',
               'uid': 'east edfa in Quimper'},
              {'metadata': {'location': {'city': 'Ploermel',
                                         'latitude': 1.0,
                                         'longitude': 2.0,
                                         'region': 'RLD'}},
               'operational': {'gain_target': None, 'tilt_target': 0},
               'type': 'Edfa',
               'uid': 'east edfa in Ploermel'},
              {'metadata': {'location': {'city': 'Lannion_CAS',
                                         'latitude': 2.0,
                                         'longitude': 0.0,
                                         'region': 'RLD'}},
               'operational': {'delta_p': 1.0,
                               'gain_target': None,
                               'out_voa': None,
                               'tilt_target': 0},
               'type': 'Edfa',
               'type_variety': 'std_medium_gain',
               'uid': 'east edfa in Lannion_CAS to Corlay'},
              {'metadata': {'location': {'city': 'Lorient_KMA',
                                         'latitude': 2.0,
                                         'longitude': 3.0,
                                         'region': 'RLD'}},
               'params': {'loss': 0},
               'type': 'Fused',
               'uid': 'east edfa in Lorient_KMA to Vannes_KBE'},
              {'metadata': {'location': {'city': 'Lannion_CAS',
                                         'latitude': 2.0,
                                         'longitude': 0.0,
                                         'region': 'RLD'}},
               'operational': {'delta_p': 1.0,
                               'gain_target': None,
                               'out_voa': None,
                               'tilt_target': 0},
               'type': 'Edfa',
               'type_variety': 'std_medium_gain',
               'uid': 'east edfa in Lannion_CAS to Stbrieuc'},
              {'metadata': {'location': {'city': 'Stbrieuc',
                                         'latitude': 1.0,
                                         'longitude': 0.0,
                                         'region': 'RLD'}},
               'operational': {'delta_p': 1.0,
                               'gain_target': None,
                               'out_voa': None,
                               'tilt_target': 0},
               'type': 'Edfa',
               'type_variety': 'std_low_gain',
               'uid': 'east edfa in Stbrieuc to Rennes_STA'},
              {'metadata': {'location': {'city': 'Lannion_CAS',
                                         'latitude': 2.0,
                                         'longitude': 0.0,
                                         'region': 'RLD'}},
               'operational': {'delta_p': 1.0,
                               'gain_target': None,
                               'out_voa': None,
                               'tilt_target': 0},
               'type': 'Edfa',
               'type_variety': 'std_medium_gain',
               'uid': 'east edfa in Lannion_CAS to Morlaix'},
              {'metadata': {'location': {'city': 'Lorient_KMA',
                                         'latitude': 2.0,
                                         'longitude': 3.0,
                                         'region': 'RLD'}},
               'operational': {'delta_p': 1.0,
                               'gain_target': None,
                               'out_voa': None,
                               'tilt_target': 0},
               'type': 'Edfa',
               'type_variety': 'std_medium_gain',
               'uid': 'east edfa in Lorient_KMA to Loudeac'},
              {'metadata': {'location': {'city': 'Vannes_KBE',
                                         'latitude': 2.0,
                                         'longitude': 4.0,
                                         'region': 'RLD'}},
               'operational': {'delta_p': 1.0,
                               'gain_target': None,
                               'out_voa': None,
                               'tilt_target': 0},
               'type': 'Edfa',
               'type_variety': 'std_medium_gain',
               'uid': 'east edfa in Vannes_KBE to Lorient_KMA'},
              {'metadata': {'location': {'city': 'Rennes_STA',
                                         'latitude': 0.0,
                                         'longitude': 0.0,
                                         'region': 'RLD'}},
               'operational': {'delta_p': 1.0,
                               'gain_target': None,
                               'out_voa': None,
                               'tilt_target': 0},
               'type': 'Edfa',
               'type_variety': 'std_medium_gain',
               'uid': 'east edfa in Rennes_STA to Stbrieuc'},
              {'metadata': {'location': {'city': 'Brest_KLA',
                                         'latitude': 4.0,
                                         'longitude': 0.0,
                                         'region': 'RLD'}},
               'operational': {'delta_p': 1.0,
                               'gain_target': None,
                               'out_voa': None,
                               'tilt_target': 0},
               'type': 'Edfa',
               'type_variety': 'std_medium_gain',
               'uid': 'east edfa in Brest_KLA to Morlaix'},
              {'metadata': {'location': {'city': 'Lannion_CAS',
                                         'latitude': 2.0,
                                         'longitude': 0.0,
                                         'region': 'RLD'}},
               'operational': {'delta_p': 1.0,
                               'gain_target': None,
                               'out_voa': None,
                               'tilt_target': 0},
               'type': 'Edfa',
               'type_variety': 'std_high_gain',
               'uid': 'west edfa in Lannion_CAS to Corlay'},
              {'metadata': {'location': {'city': 'Lorient_KMA',
                                         'latitude': 2.0,
                                         'longitude': 3.0,
                                         'region': 'RLD'}},
               'operational': {'delta_p': 1.0,
                               'gain_target': None,
                               'out_voa': None,
                               'tilt_target': 0},
               'type': 'Edfa',
               'type_variety': 'std_low_gain',
               'uid': 'west edfa in Lorient_KMA to Vannes_KBE'},
              {'metadata': {'location': {'city': 'Lannion_CAS',
                                         'latitude': 2.0,
                                         'longitude': 0.0,
                                         'region': 'RLD'}},
               'operational': {'delta_p': 1.0,
                               'gain_target': None,
                               'out_voa': None,
                               'tilt_target': 0},
               'type': 'Edfa',
               'type_variety': 'std_low_gain',
               'uid': 'west edfa in Lannion_CAS to Stbrieuc'},
              {'metadata': {'location': {'city': 'Stbrieuc',
                                         'latitude': 1.0,
                                         'longitude': 0.0,
                                         'region': 'RLD'}},
               'operational': {'delta_p': 1.0,
                               'gain_target': None,
                               'out_voa': None,
                               'tilt_target': 0},
               'type': 'Edfa',
               'type_variety': 'std_low_gain',
               'uid': 'west edfa in Stbrieuc to Rennes_STA'},
              {'metadata': {'location': {'city': 'Lannion_CAS',
                                         'latitude': 2.0,
                                         'longitude': 0.0,
                                         'region': 'RLD'}},
               'operational': {'delta_p': 1.0,
                               'gain_target': None,
                               'out_voa': None,
                               'tilt_target': 0},
               'type': 'Edfa',
               'type_variety': 'std_low_gain',
               'uid': 'west edfa in Lannion_CAS to Morlaix'},
              {'metadata': {'location': {'city': 'Lorient_KMA',
                                         'latitude': 2.0,
                                         'longitude': 3.0,
                                         'region': 'RLD'}},
               'operational': {'delta_p': 1.0,
                               'gain_target': None,
                               'out_voa': None,
                               'tilt_target': 0},
               'type': 'Edfa',
               'type_variety': 'std_high_gain',
               'uid': 'west edfa in Lorient_KMA to Loudeac'},
              {'metadata': {'location': {'city': 'Vannes_KBE',
                                         'latitude': 2.0,
                                         'longitude': 4.0,
                                         'region': 'RLD'}},
               'operational': {'delta_p': 1.0,
                               'gain_target': None,
                               'out_voa': None,
                               'tilt_target': 0},
               'type': 'Edfa',
               'type_variety': 'std_medium_gain',
               'uid': 'west edfa in Vannes_KBE to Lorient_KMA'},
              {'metadata': {'location': {'city': 'Rennes_STA',
                                         'latitude': 0.0,
                                         'longitude': 0.0,
                                         'region': 'RLD'}},
               'operational': {'delta_p': 1.0,
                               'gain_target': None,
                               'out_voa': None,
                               'tilt_target': 0},
               'type': 'Edfa',
               'type_variety': 'std_low_gain',
               'uid': 'west edfa in Rennes_STA to Stbrieuc'},
              {'metadata': {'location': {'city': 'Brest_KLA',
                                         'latitude': 4.0,
                                         'longitude': 0.0,
                                         'region': 'RLD'}},
               'operational': {'delta_p': 1.0,
                               'gain_target': None,
                               'out_voa': None,
                               'tilt_target': 0},
               'type': 'Edfa',
               'type_variety': 'std_low_gain',
               'uid': 'west edfa in Brest_KLA to Morlaix'}]}
    
    network_02 = network_from_json(new_nw_definition,equipment)
    print(type(network_02))
    path_request_03 = create_path_request(equipment, req_params_mesh)
    path,infos, infos_02, si_list_02, infos_03, si_list_03 = simulate(equipment, network_02, path_request_03)
    print_path(path)
    x = final_gsnr(path)
    print('Final GSNR in dB')
    print(x)
    #network = load_net(topology_path, equipment)

elif number_of_simulation == 'First-Example':
    '''
    1. Step: Define Simulation Input
        1.1. Load Network-Equipment
        1.2. Load Network-Topology
        1.3. Load Path-Request
    2. Step: Define Simulation Settings
    3. Step: Run Simulation
    '''
    #import additional important functions from GNPyWrapper
    from GNPyWrapper import get_from_storage_of_network_equipment
    from GNPyWrapper import get_from_storage_of_network_topologies
    from GNPyWrapper import get_from_storage_of_path_requests
    #display which simulation is running:
    print('The following simulation is running:' + str(number_of_simulation))
    #1. Step: Define Simulation Input
    #1.1 Load Network-Equipment
    #Network-equipment in python dictionary format
    network_equipment_dict = get_from_storage_of_network_equipment('example_01')     
    #create a random filename
    filename = 'random-filename' # this is necessary for Amp.from_json function...
    # do gnpy stuff with it...
    nw_equipment= _equipment_from_json(network_equipment_dict, filename)
    #1.2 Load Network topology
    # network-topology-dict
    network_topology_dict= get_from_storage_of_network_topologies('example_01')    
    #create Di.Graph from python dictionary for network-topology using the provided equipment
    nw_topology = network_from_json(network_topology_dict,nw_equipment)
    # 1.3 Load Path-Request
    # define path-request in python dict format
    path_request_dict_mesh = get_from_storage_of_path_requests('example_01')
    #create a path request using the path request dict and the equipment
    path_request = create_path_request(nw_equipment, path_request_dict_mesh)
    #print(type(path_request))
    #pprint.pprint(path_request)
    #print(path_request.__str__())
    #print(path_request.__repr__())
    #3. Simulate
    path,infos, infos_02, si_list_02, infos_03, si_list_03 = simulate(nw_equipment, nw_topology, path_request)
    print_path(path)
    x = final_gsnr(path)
    print('Final GSNR in dB')
    print(x)
else:
    print('no simulation was picked')
