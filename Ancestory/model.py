import os
import pandas as pd

import Ancestory.tree as tree


SCHEMA = {
    'patient': {'columns': ["Patient_ID", "Name", "Age", "Is_Dead"], 'index' :'Patient_ID'},
    'child': {'columns': ["Patient_ID", "Mother_ID", "Father_ID"], 'index': 'Patient_ID'},
    'disease': {'columns': ["Disease_ID", "Disease_name"], 'index': 'Disease_ID'},
    'patient_disease': {'columns':['PersonDisease_ID','Patient_ID','Disease_ID'], 'index': 'PersonDisease_ID'}
}

TREE_SCHEMA = {'mother': 'Mother_ID', 'father': 'Father_ID', 'child': 'Patient_ID'}

def load_df(path, columns, write_if_empty=True):
    if os.path.isfile(path):
        return pd.read_csv(path)
    created_df = pd.DataFrame(columns=columns)
    if write_if_empty:
        created_df.to_csv(path)
    return created_df


class AncestoryModel:

    def set_indexes(self):
        for key in self.data:
            id = SCHEMA[key]['index']
            data = self.data[key]['data']
            if id is not None:
                data.set_index(id, inplace=True)
    
    def fetch_data(self, df_key):
        return self.data[df_key]['data'].copy()
    
    def diseases(self, patient_id):
        patient_disease = self.data['patient_disease']['data']
        if patient_id is None:
            return self.fetch_data('disease').index.tolist()
        return patient_disease.loc[patient_disease.Patient_ID==patient_id, 'Disease_ID'].tolist()
    
    def name(self, patient_id):
        if pd.isna(patient_id):
            return None
        patient = self.data['patient']['data']
        return patient.loc[patient_id, 'Name']
    
    def count_relatives_with_disease(self, patient_id, disease_id):
        # Step 1: Check if the patient themselves has the disease
        patient_has_disease = disease_id in self.diseases(patient_id)

        patient_disease_df = self.data['patient_disease']['data']
        child_df = self.data['child']['data']

        siblings, grandchildren, parents, children = tree.find_siblings_grandchildren(patient_id, child_df.reset_index(), TREE_SCHEMA, True)

        # Combine all relatives into sets to avoid duplication
        primary_relatives = set(parents).union(set(children))
        secondary_relatives = set(siblings).union(set(grandchildren))

        # Step 4: Filter relatives who have the given disease
        diseased_relatives = patient_disease_df[patient_disease_df["Disease_ID"] == disease_id]["Patient_ID"]

        # Count primary and secondary relatives with the disease
        primary_with_disease = len(primary_relatives.intersection(diseased_relatives))
        secondary_with_disease = len(secondary_relatives.intersection(diseased_relatives))

        # Include the patient in the final count if they have the disease
        total_with_disease = primary_with_disease + secondary_with_disease

        result = pd.Series({
            "Does the patient have the disease?": 'YES' if patient_has_disease else 'NO',
            "Number of primary degree relatives with disease": primary_with_disease,
            "Number of secondary degree relatives with disease": secondary_with_disease,
            "Total number of relatives with disease": total_with_disease
        })

        verdict = 'Low Risk'
        
        if patient_has_disease:
            verdict = ':red[Get treatment soon]'
        elif primary_with_disease > 0:
            verdict = ':red[Strong] reason to investigate'
        elif secondary_with_disease > 0:
            verdict = ':orange[Possible] reason to investigate'
        else:
            verdict = ':green[Low risk]'

        return result, verdict
    

    def people(self):
        return self.data['patient']['data'].index.to_list()

    def disease_name(self, disease_id):
        if pd.isna(disease_id):
            return None
        disease = self.data['disease']['data']
        return disease.loc[disease_id, 'Disease_name']
    
    def get_edge_list(self):
        df = self.data['child']['data']
        return tree.convert_child_df_to_edge_list(df.reset_index(), schema=TREE_SCHEMA)
    
    def find_descendants(self, patient_id):
        df = self.data['child']['data']
        return tree.find_descendants(patient_id, df.reset_index(), schema=TREE_SCHEMA)

    def find_descendants_by_level(self, patient_id):
        df = self.data['child']['data']
        return tree.find_descendants_by_level(patient_id, df.reset_index(), schema=TREE_SCHEMA)
    
    def find_ancestors_by_level(self, patient_id):
        df = self.data['child']['data']
        return tree.find_ancestors_by_level(patient_id, df.reset_index(), schema=TREE_SCHEMA)

    def filter_family_tree(self, patient_id, as_edge_list=False):
        df = self.data['child']['data']
        family_tree = tree.filter_family_tree(df.reset_index(), patient_id, schema=TREE_SCHEMA)
        if as_edge_list:
            return tree.convert_child_df_to_edge_list(family_tree, schema=TREE_SCHEMA)
        return family_tree

    def add_record(self, df_key, record_dict, update_in_file=True):
        entry = self.data[df_key]
        data, path = entry['data'], entry['path']
        index_col = SCHEMA[df_key]['index']
        if index_col is None:
            # index can be assumed to be numercial 
            index = data.index.max() + 1
            record_list = [record_dict[v] for v in SCHEMA[df_key]['columns']]
        else:
            record_list = [record_dict[v] for v in SCHEMA[df_key]['columns'] if index_col!=v]
            index = record_dict[index_col] 
        data.loc[index] = record_list
        # Update data in path:
        if update_in_file:
            data.to_csv(path)
    
    def add_records(self, df_key, record_dict_list):
        N = len(record_dict_list)
        for i in range(N):
            record_dict = record_dict_list[i]
            self.add_record(df_key, record_dict, i==(N-1))

    def update_patient(self, id, name, age, is_dead):
        patient_record = {"Patient_ID": id, "Name": name, "Age": age, "Is_Dead": is_dead}
        self.add_record('patient', patient_record)

    def update_child(self, patient_id, mother_id, father_id):
        child_record = {"Patient_ID": patient_id, "Mother_ID": mother_id, "Father_ID": father_id}
        self.add_record('child', child_record)

    def fetch_parents(self, child_id):
        df = self.data['child']['data']
        if child_id not in df.index:
            return None
        res = self.data['child']['data'].loc[child_id]
        return res
    
    def fetch_mother(self, child_id):
        parents = self.fetch_parents(child_id)
        if parents is not None and 'Mother_ID' in parents:
            return parents.Mother_ID
        return None
    
    def fetch_father(self, child_id):
        parents = self.fetch_parents(child_id)
        if parents is not None and 'Father_ID' in parents:
            return parents.Father_ID
        return None
    
    def update_disease(self, disease_ID,disease_name):
        disease_record = {'Disease_ID': disease_ID, 'Disease_name': disease_name}
        self.add_record('disease', disease_record)
        return disease_record
    
    def get_max_id(self, df_key):
        entry = self.data[df_key]['data']
        return entry.index[len(entry.index)-1].max()
    
    def drop_if_column(self, df_key, column_to_search, val_in_col):
        entry = self.data[df_key]
        data = entry['data']
        dropped = data.loc[~(data[column_to_search]==val_in_col)]
        self.data[df_key]['data'] = dropped
        dropped.to_csv(entry['path'])
  
    def add_patient_disease(self, patient_disease_tuples):
        last_id = self.get_max_id('patient_disease')
        id = last_id + 1
        records = [{'Patient_ID': patient_disease_tuples[i][0], 
                    'Disease_ID': patient_disease_tuples[i][1],
                    'PersonDisease_ID': id + i} for i in range(len(patient_disease_tuples))]
        self.add_records('patient_disease', records)
 
    def update_diseases_for_patient(self, patient_id, disease_ids):
        self.drop_if_column('patient_disease', 'Patient_ID', patient_id)
        N = len(disease_ids)
        patient_disease_tuples = list(zip([patient_id]*N, disease_ids))
        self.add_patient_disease(patient_disease_tuples)
 
    def __init__(self, patient_table_path, child_table_path, disease_table_path, patient_disease_table_path):
        patient_df = load_df(patient_table_path, columns=SCHEMA['patient']['columns'], write_if_empty=True)
        child_df = load_df(child_table_path, columns=SCHEMA['child']['columns'], write_if_empty=True)
        disease_df = load_df(disease_table_path, columns=SCHEMA['disease']['columns'], write_if_empty=True)
        patient_disease_df = load_df(patient_disease_table_path, columns=SCHEMA['patient_disease']['columns'], write_if_empty=True)

        self.data = {'patient': {'data': patient_df, 'path': patient_table_path},
                     'child': {'data': child_df, 'path': child_table_path},
                     'disease': {'data': disease_df, 'path': disease_table_path},
                     'patient_disease': {'data': patient_disease_df, 'path': patient_disease_table_path}
                     }
        
        self.set_indexes()


