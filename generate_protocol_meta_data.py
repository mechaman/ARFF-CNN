import os
import numpy as np
import nibabel as nib
from pathlib import Path
from utils import load_partitioned_data
import pandas as pd
import re

def extract_protocol(patient):
    '''
    Constructs the protocol from voxel dim. & voxel size
    '''
    patient_data = nib.load(patient)
    patient_header = patient_data.header
    # Extract voxel dims. & voxel size
    voxel_dims = patient_header.get_data_shape()
    voxel_size = patient_header.get_zooms()
    # Create protocol protocol = voxel dims + voxel size
    protocol_tup = voxel_dims + voxel_size
    # Construct protocol string
    protocol = ','.join(list(map(str, protocol_tup)))
    return protocol
    
def load_patients(fp):
    '''
    Returns patient file names
    '''
    partition = {}
    (_,
     patient_names,
     _,
     _,
     _,
     _) = load_partitioned_data(fp, split=(100, 0, 0), y_label='_defaced')
    return patient_names 

def parse_patient_name(patient_fp):
    '''
    Parse the patient name from the fp
    '''
    parsed_patient = re.search('\/(.*)\/(.*)_defaced.nii', patient_fp)
    if parsed_patient is None:
        return None
    patient_name = parsed_patient.group(2)
    return patient_name

def generate_p2p_csv(fp):
    ''' 
    Generates/Updates csv w/ patient -> protocol
    '''
    # Dataframe containing meta data
    p2p = None
    meta_data_exists = False
    # Load Patients
    data_fp = './data'
    patient_fps = load_patients(data_fp)
    # Check if output csv exists
    if Path(fp).is_file():
        print('Output csv exists... Loading...')
        p2p = pd.read_csv(fp).set_index('Patient')
        meta_data_exists = True
    else: 
        col_names = ['Patient', 'Protocol']
        p2p = pd.DataFrame(columns = col_names).set_index('Patient')

    # Iterate through patients in data set
    for patient_fp in patient_fps:
        patient_name = parse_patient_name(patient_fp)
        # Check if parsing worked
        if patient_name is None:
            print('patient_fp does not conform.')
            continue
        else:
            # Add patient if doesn't exist
            if patient_name not in p2p.index:
                patient_protocol = extract_protocol(patient_fp)
                # Add row with patient protocol
                p2p_temp = pd.DataFrame(data={'Patient':[patient_name], 'Protocol':[patient_protocol]}).set_index('Patient')
                p2p = pd.concat([p2p, p2p_temp])
            else:
                continue
        
    ## Write CSV 
    p2p.to_csv(fp) 
    

    
        


     

if __name__ == "__main__":
    output_fp = './patient_meta.csv'
    generate_p2p_csv(fp = output_fp) 
