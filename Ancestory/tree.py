import pandas as pd
# Contains all the methods required for processing familial relations in a family tree structure
# Defines family structure as a pandas dataframe with columns: [Mother_ID, Father_ID, Patient_ID]
# Schema can be manually inputted in each method to change default schema 

SCHEMA = {'mother': 'Mother_ID', 'father': 'Father_ID', 'child': 'Patient_ID'}

# Given a dataframe (pandas) with a list of (mother, father, child) records:
# find_descendants will recursively find the descendants for input parent_id
def find_descendants(parent_id, df, schema=SCHEMA):
    # Initialize a set to store all descendants
    descendants = set()
    # Recursive function to find children and their descendants
    def find_children(parent_id):
        # Find children where the parent_id matches either Mother_ID or Father_ID
        children = df[(df[schema['mother']] == parent_id) | (df[schema['father']] == parent_id)][schema['child']]
        # For each child, add to descendants and find their children recursively
        for child in children:
            if child not in descendants:
                descendants.add(child)
                find_children(child)  # Recursively find this child's descendants
    # Start finding descendants from the given parent_id
    find_children(parent_id)
    return list(descendants)


# Given a dataframe (pandas) with a list of (mother, father, child) records:
# find_ancestors will recursively find the ancestors up to input level for input child_id
def find_ancestors_by_level(df, child_id, levels=2, schema=SCHEMA):
    """Find all ancestors (parents, grandparents) up to `levels`."""
    ancestors = set()
    def dfs_ancestors(pid, level):
        if level > levels:
            return
        # Find the row with the matching patient
        row = df[df[schema['child']] == pid]
        if not row.empty:
            mother = row.iloc[0][schema['mother']]
            father = row.iloc[0][schema['father']]
            if pd.notna(mother):
                ancestors.add(mother)
                dfs_ancestors(mother, level + 1)
            if pd.notna(father):
                ancestors.add(father)
                dfs_ancestors(father, level + 1)

    dfs_ancestors(child_id, 1)
    return ancestors

# Given a dataframe (pandas) with a list of (mother, father, child) records:
# find_ancestors will recursively find the descendants up to input level for input child_id
def find_descendants_by_level(df, child_id, levels=2, schema=SCHEMA):
    """Find all descendants (children, grandchildren) up to `levels`."""
    descendants = set()
    def dfs_descendants(pid, level):
        if level > levels:
            return
        # Find all rows where the given pid is listed as a mother or father
        children = df[(df[schema['mother']] == pid) | (df[schema['father']] == pid)][schema['child']]
        for child in children:
            descendants.add(child)
            dfs_descendants(child, level + 1)
    dfs_descendants(child_id, 1)
    return descendants


# Will filter the given family tree to just the family tree of the input node (child_id)
def filter_family_tree(df, child_id, schema=SCHEMA):
    """Filter the DataFrame to only include the patient, their ancestors (2 levels up), and descendants (2 levels down)."""
    # Find all relevant nodes
    ancestors = find_ancestors_by_level(df, child_id, schema=schema)
    descendants = find_descendants_by_level(df, child_id, schema=schema)
    # Include the patient itself
    relevant_nodes = ancestors | descendants | {child_id}
    # Filter the DataFrame to only include the relevant nodes
    filtered_df = df[df[schema['child']].isin(relevant_nodes)]
    return filtered_df


def find_parents_children(patient_id, child_df, schema=SCHEMA):
    child, mother, father = schema['child'], schema['mother'], schema['father']
    parents = child_df[(child_df[child] == patient_id)][[mother, father]].values.flatten()
    children = child_df[(child_df[mother] == patient_id) | (child_df[father] == patient_id)][child]
    return parents, children


def find_primary_degree_relatives(patient_id, child_df, schema=SCHEMA):
    parents, children = find_parents_children(patient_id, child_df, schema)
    primary_relatives = set(parents).union(set(children))
    return primary_relatives


def find_siblings_grandchildren(patient_id, child_df, schema=SCHEMA, include_parents_children=False):
    child, mother, father = schema['child'], schema['mother'], schema['father']
    parents, children = find_parents_children(patient_id, child_df)
    # Find secondary relatives (siblings and grandchildren)
    siblings = child_df[(child_df[mother].isin(parents)) & 
                        (child_df[father].isin(parents))][child]
    grandchildren = child_df[child_df[mother].isin(children) | child_df[father].isin(children)][child]
    if not include_parents_children:
        return siblings, grandchildren
    return siblings, grandchildren, parents, children


def find_secondary_degree_relatives(patient_id, child_df, schema=SCHEMA):
    siblings, grandchildren = find_siblings_grandchildren(patient_id, child_df, schema)
    secondary_relatives = set(siblings).union(set(grandchildren))
    return secondary_relatives



def convert_child_df_to_edge_list(child_df, schema=SCHEMA):
    child, mother, father = schema['child'], schema['mother'], schema['father']

    # Convert the Child_df  to a list of dictionaries
    linking_data = child_df.to_dict(orient='records')

    rows = []
    # Iterate through each dictionary in the linking_data list
    for entry in linking_data:
        patient_id = entry[child]
        
        # Add a row for the mother relationship
        if mother in entry:
            mid = entry[child]
            if pd.notna(mid):
                rows.append({
                    "source": entry[mother],
                    "target": patient_id,
                    "relationship": "mother"
                })
        
        # Add a row for the father relationship
        if father in entry:
            fid = entry[father]
            if pd.notna(fid):
                rows.append({
                    "source": entry[father],
                    "target": patient_id,
                    "relationship": "father"
                })
    
    # Create a DataFrame from the rows list
    linking_df = pd.DataFrame(rows)
    
    return linking_df




# Function to find relatives with a specific disease
def count_relatives_with_disease(patient_id, disease_id, child_df, patient_disease_df):
    # Step 1: Check if the patient themselves has the disease
    patient_has_disease = patient_id in patient_disease_df[
        patient_disease_df["Disease_ID"] == disease_id
    ]["Patient_ID"].values

    # Step 2: Find primary relatives (parents and children)
    parents = child_df[(child_df["Patient_ID"] == patient_id)][["Mother_ID", "Father_ID"]].values.flatten()
    children = child_df[(child_df["Mother_ID"] == patient_id) | (child_df["Father_ID"] == patient_id)]["Patient_ID"]

    # Step 3: Find secondary relatives (siblings and grandchildren)
    siblings = child_df[(child_df["Mother_ID"].isin(parents)) & 
                        (child_df["Father_ID"].isin(parents))]["Patient_ID"]
    grandchildren = child_df[child_df["Mother_ID"].isin(children) | child_df["Father_ID"].isin(children)]["Patient_ID"]

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

    patient_count = 0
    if patient_has_disease:
        patient_count = 1
        total_with_disease += 0

    primary_with_disease = primary_with_disease+patient_count
    secondary_with_disease = secondary_with_disease - patient_count

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



