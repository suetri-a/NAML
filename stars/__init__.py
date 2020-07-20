# Utilities for interfacing with CMG-STARS

import os
import numpy as np

from .rto.reactions import Component, Kinetics

BASE_MODEL_DICT = {'CINAR': 1, 'KRISTENSEN': 2, 'CHEN': 3}


def get_component_dict(comp_names, special_model = None):
    '''
    Inputs:
        comp_names - names of components used in the model
        special_model - model to be used other than the default to indicate different units

    Returns:
        component_dict - dictionary of Component() class entries for each component in comp_names
    '''

    component_dict = {}

    if special_model is None:

        ##### DEFAULT COMPONENTS FOR ALL RTO SIMULATIONS
        component_dict["H2O"] = Component(COMPNAME="H2O",CMM=[1.8e-2],PCRIT=[22083],TCRIT=[373.8],
                                        AVG=[0],BVG=[0],AVISC=[0],BVISC=[0],
                                        CPG1=[0],CPG2=[0],CPG3=[0],CPG4=[0],CPL1=[0],CPL2=[0],
                                        MASSDEN=[0.001],CP=[0],CT1=[0],CT2=[0],SOLID_DEN=[],SOLID_CP=[],
                                        HVAPR=[0], EV=[0],
                                        phase=1)
        component_dict["Oil"] = Component(COMPNAME="Oil",CMM=[9.26e-1],PCRIT=[890],TCRIT=[171],
                                        AVG=[0.0001610891804],BVG=[0.7453161006],AVISC=[1.426417368e-11],BVISC=[10823.06574],
                                        CPG1=[26.804420692906],CPG2=[0.005649089963],CPG3=[0.000095012314],CPG4=[-0.000000054709],
                                        CPL1=[524.8821790],CPL2=[1.148635444845],
                                        MASSDEN=[0.000999798],CP=[7.25e-7],CT1=[0.00069242],CT2=[0],
                                        HVAPR=[0], EV=[0],
                                        phase=2)
        component_dict["N2"] = Component(COMPNAME="N2",CMM=[2.8e-2],
                                        PCRIT=[3392],TCRIT=[-147],
                                        AVG=[0.0003500869287],BVG=[0.6927470725],AVISC=[],BVISC=[],
                                        CPG1=[30.956477056957],CPG2=[-0.012716023994],CPG3=[0.000025490143],CPG4=[-0.000000011065],
                                        CPL1=[0],CPL2=[0],
                                        phase=3)
        component_dict["O2"] = Component(COMPNAME="O2",CMM=[3.2e-2],
                                        PCRIT=[5033],TCRIT=[-118],
                                        AVG=[0.000362791571],BVG=[0.7120986013],AVISC=[],BVISC=[],
                                        CPG1=[28.600167325729],CPG2=[-0.003497011859],CPG3=[0.000024399453],CPG4=[-0.000000014928],
                                        CPL1=[0],CPL2=[0],
                                        phase=3)
        component_dict["CO2"] = Component(COMPNAME="CO2",CMM=[4.4e-2],
                                        PCRIT=[7377],TCRIT=[31],
                                        AVG=[0.0001865724378],BVG=[0.7754816784],AVISC=[],BVISC=[],
                                        CPG1=[19.474325955388],CPG2=[0.075654731286],CPG3=[-0.000060750197],CPG4=[0.000000020109],
                                        CPL1=[0],CPL2=[0],
                                        phase=3)
        component_dict["CO"] = Component(COMPNAME="CO",CMM=[2.8e-2],
                                        PCRIT=[3496],TCRIT=[-144],
                                        AVG=[0.0003315014585],BVG=[0.7037315714],AVISC=[],BVISC=[],
                                        CPG1=[30.990187019402],CPG2=[-0.01392019971],CPG3=[0.00003014996],CPG4=[-0.00000001415],
                                        CPL1=[0],CPL2=[0],
                                        phase=3)

        ##### GENERATE PSEUDOCOMPONENT ENTRIES
        for c in comp_names:
            if c not in component_dict.keys():
                if c=='Gas':
                    component_dict["Gas"] = Component(COMPNAME="Gas",CMM=[2.8e-2],
                                        PCRIT=[3496],TCRIT=[-144],
                                        AVG=[0.0003315014585],BVG=[0.7037315714],AVISC=[],BVISC=[],
                                        CPG1=[30.990187019402],CPG2=[-0.01392019971],CPG3=[0.00003014996],CPG4=[-0.00000001415],
                                        CPL1=[0],CPL2=[0],
                                        phase=3)
                elif len(c)>=3:
                    if c[:3]=='Oil':
                        component_dict[c] =  Component(COMPNAME=c,CMM=[9.26e-1],PCRIT=[890],TCRIT=[171],
                                                        AVG=[0.0001610891804],BVG=[0.7453161006],AVISC=[1.426417368e-11],BVISC=[10823.06574],
                                                        CPG1=[26.804420692906],CPG2=[0.005649089963],CPG3=[0.000095012314],CPG4=[-0.000000054709],
                                                        CPL1=[524.8821790],CPL2=[1.148635444845],
                                                        MASSDEN=[0.000999798],CP=[7.25e-7],CT1=[0.00069242],CT2=[0],
                                                        HVAPR=[0], EV=[0],
                                                        phase=2)
                    elif c[:4]=='Coke':
                        component_dict[c] = Component(COMPNAME=c,CMM=[1.36e-2],
                                                        SOLID_DEN=[0.0014,0,0],SOLID_CP=[6.96015,0.03645],
                                                        phase=4)
                    else:
                        raise Exception('Invalid pseudocomponent {c} entered.'.format(c=c))
                else:
                    raise Exception('Invalid pseudocomponent {c} entered.'.format(c=c))


    else:
        raise Exception('Invalid model {m} entered'.format(m=special_model))


    return component_dict