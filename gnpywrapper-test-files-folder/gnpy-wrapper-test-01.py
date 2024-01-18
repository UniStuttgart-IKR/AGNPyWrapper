## adding the necessary packages to the python path
# import sys
# for networkx and matplotlib
# sys.path.append('...')

# for gnpywrapper
# sys.path.append('...')

# for gnpy
# sys.path.append(...)

import networkx
# create graph
g = networkx.DiGraph()

# Steps:
#     1. Define configuration of element
#     2. create object of class/specific element
#     3. add element to graph as node
#     4. add edges
#     5. define signal
#     6. define path
#     7. propagate signal on path
#    (8. Analyse results)
from gnpywrapper.example_storage import get_from_storage_of_network_element_configurations
from gnpy.core import elements

# first element
el_config = get_from_storage_of_network_element_configurations('Transceiver', '1')
el_01 = elements.Transceiver(**el_config)
g.add_node(el_01)


import matplotlib.pyplot as plt
# show graph
networkx.draw_planar(g, with_labels=True)
plt.show()

# second element
el_config = get_from_storage_of_network_element_configurations('Transceiver', '2')
el_02 = elements.Transceiver(**el_config)
g.add_node(el_02)

# show graph
networkx.draw_planar(g, with_labels=True)
plt.show()

# 3. element
el_config = get_from_storage_of_network_element_configurations('Roadm', '1')
el_03 = elements.Roadm(**el_config)
g.add_node(el_03)

# show graph
networkx.draw_planar(g, with_labels=True)
plt.show()

# 4. element
el_config = get_from_storage_of_network_element_configurations('Roadm', '2')
el_04 = elements.Roadm(**el_config)
g.add_node(el_04)

# show graph
networkx.draw_planar(g, with_labels=True)
plt.show()

# 5. element
el_config = get_from_storage_of_network_element_configurations('Fiber', '1')
el_05 = elements.Fiber(**el_config)
g.add_node(el_05)

# show graph
networkx.draw_planar(g, with_labels=True)
plt.show()

# 6. element
el_config = get_from_storage_of_network_element_configurations('Edfa', '1b')
el_06 = elements.Edfa(**el_config)
g.add_node(el_06)

# show graph
networkx.draw_planar(g, with_labels=True)
plt.show()

# 7. element
el_config = get_from_storage_of_network_element_configurations('Edfa', '2')
el_07 = elements.Edfa(**el_config)
g.add_node(el_07)

# show graph
networkx.draw_planar(g, with_labels=True)
plt.show()

# now add edges:
# it is possible to directly make edges with these elements

# add 1. edge
g.add_edge(el_01,el_03)

# show graph
networkx.draw_planar(g, with_labels=True)
plt.show()


# add 2. edge
g.add_edge(el_03,el_06)

# show graph
networkx.draw_planar(g, with_labels=True)
plt.show()


# add 3. edge
g.add_edge(el_06,el_05)

# show graph
networkx.draw_planar(g, with_labels=True)
plt.show()


# add 4. edge
g.add_edge(el_05,el_07)

# show graph
networkx.draw_planar(g, with_labels=True)
plt.show()

# add 5. edge
g.add_edge(el_07,el_04)

# show graph
networkx.draw_planar(g, with_labels=True)
plt.show()

# add 6. edge
g.add_edge(el_04,el_02)

# show graph
networkx.draw_planar(g, with_labels=True)
plt.show()

# define path on which signal should propagate
path_generator= networkx.shortest_simple_paths(g,el_01,el_02)
paths = list(path_generator)
#print(paths)

import pprint
pprint.pprint(paths)
# pick shortest path 
path = paths[0]


# Define Signal which should propagate on path
from gnpywrapper.example_storage import get_from_storage_of_spectral_information

si_dictionary = get_from_storage_of_spectral_information('example_1')

from gnpy.core.info import SpectralInformation
si_start = SpectralInformation(frequency=si_dictionary['frequency'], 
                        slot_width=si_dictionary['slot_width'],
                        signal= si_dictionary['signal'], 
                        nli= si_dictionary['nli'], 
                        ase= si_dictionary['ase'],
                        baud_rate=si_dictionary['baud_rate'], 
                        roll_off=si_dictionary['roll_off'],
                        chromatic_dispersion= si_dictionary['chromatic_dispersion'],       
                        pmd=si_dictionary['pmd'], 
                        pdl=si_dictionary['pdl'], 
                        latency=si_dictionary['latency'],       
                        delta_pdb_per_channel=si_dictionary['delta_pdb_per_channel'],
                        tx_osnr= si_dictionary['tx_osnr'],
                        ref_power= si_dictionary['ref_power'], 
                        label= si_dictionary['label'])


#propagate signal on path
from gnpywrapper.functions import propagate_with_update_snr
si_result = propagate_with_update_snr(path,si_start)




# show graph
networkx.draw_planar(g, with_labels=True)
plt.show()


# show path (results) (using the represent functions of gnpy elements)
# print results
# for elem in path:
#     print(elem)