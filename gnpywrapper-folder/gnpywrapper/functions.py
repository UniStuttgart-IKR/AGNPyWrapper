# import important classes and functions from gnpy
from gnpy.core.info import SpectralInformation
from gnpy.core.elements import Roadm


def propagate(path,si): # by jensk
    """propagates signals in each element according to initial spectrum set by user"""
    for i, el in enumerate(path):
        if isinstance(el, Roadm):
            si = el(si, degree=path[i+1].uid)
        else:
            si = el(si)
            
    # here update_snr-step is missing.
    # this is intended
    # there is also another function which those the same things like this function but additionally is considers the update_snr step  
    return si

def propagate_with_update_snr(path,si): # by jensk
    """propagates signals in each element according to initial spectrum set by user"""
    for i, el in enumerate(path):
        if isinstance(el, Roadm):
            si = el(si, degree=path[i+1].uid)
        else:
            si = el(si)
    
    #update_snr - step:
    
    DEFAULT_ADD_DROP_OSNR_BY_ROADM = 33
    
    path[0].update_snr(si.tx_osnr)
    # path[0].calc_penalties(req.penalties)  # has no influence (as I see it right now (14.01.2024) by jensk) 
    if any(isinstance(el, Roadm) for el in path):
        path[-1].update_snr(si.tx_osnr, DEFAULT_ADD_DROP_OSNR_BY_ROADM)
    else:
        path[-1].update_snr(si.tx_osnr)
    #path[-1].calc_penalties(req.penalties)      # has no influence (as I see it right now (14.01.2024) by jensk)   
            

    return si
    
def propagate_save_history(path,si): # by jensk
    """propagates signals in each element according to initial spectrum set by user"""
    list_of_si_objects =[]
    for i, el in enumerate(path):
        if isinstance(el, Roadm):
            si = el(si, degree=path[i+1].uid)
        else:
            si = el(si)
        
        u=SpectralInformation(frequency=si._frequency, 
                            slot_width=si._slot_width,
                            signal=si._signal, 
                            nli=si._nli, 
                            ase=si._ase,
                            baud_rate=si._baud_rate, 
                            roll_off=si._roll_off,
                            chromatic_dispersion=si._chromatic_dispersion,       
                            pmd=si._pmd, 
                            pdl=si._pdl, 
                            latency=si._latency,       
                            delta_pdb_per_channel=si._delta_pdb_per_channel,
                            tx_osnr=si._tx_osnr,
                            ref_power=si._pref, 
                            label=si._label)
        
        list_of_si_objects.append(u)

    return si,list_of_si_objects

def propagate_save_history_with_update_snr(path,si): # by jensk
    """propagates signals in each element according to initial spectrum set by user"""
    list_of_si_objects =[]
    for i, el in enumerate(path):
        if isinstance(el, Roadm):
            si = el(si, degree=path[i+1].uid)
        else:
            si = el(si)
        
        u=SpectralInformation(frequency=si._frequency, 
                            slot_width=si._slot_width,
                            signal=si._signal, 
                            nli=si._nli, 
                            ase=si._ase,
                            baud_rate=si._baud_rate, 
                            roll_off=si._roll_off,
                            chromatic_dispersion=si._chromatic_dispersion,       
                            pmd=si._pmd, 
                            pdl=si._pdl, 
                            latency=si._latency,       
                            delta_pdb_per_channel=si._delta_pdb_per_channel,
                            tx_osnr=si._tx_osnr,
                            ref_power=si._pref, 
                            label=si._label)
        
        list_of_si_objects.append(u)
        
    #update_snr - step:
    
    DEFAULT_ADD_DROP_OSNR_BY_ROADM = 33
    
    path[0].update_snr(si.tx_osnr)
    # path[0].calc_penalties(req.penalties)  # has no influence (as I see it right now (14.01.2024) by jensk) 
    if any(isinstance(el, Roadm) for el in path):
        path[-1].update_snr(si.tx_osnr, DEFAULT_ADD_DROP_OSNR_BY_ROADM)
    else:
        path[-1].update_snr(si.tx_osnr)
    #path[-1].calc_penalties(req.penalties)      # has no influence (as I see it right now (14.01.2024) by jensk)  

    return si,list_of_si_objects