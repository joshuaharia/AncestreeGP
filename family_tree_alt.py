import streamlit as st
import networkx as nx
import nx_altair as nxa
import altair as alt
import pandas as pd
import pydot
from networkx.drawing.nx_pydot import graphviz_layout



import os

patient_path = os.path.join('Data', 'patients.csv')
child_path = os.path.join('Data', 'child.csv')
disease_path = os.path.join('Data', 'disease.csv')
patient_disease_path = os.path.join('Data', 'patient_disease.csv')

def load_df(path, empty_df=None):
    if os.path.isfile(path):
        return pd.read_csv(path)
    return empty_df


def load_from_path():
    patient_df = load_df(patient_path, pd.DataFrame(columns=["Patient_ID", "Name", "Age", "Is_Dead"]))
    patient_df.set_index('Patient_ID', inplace=True)
    patient_df.to_csv(patient_path)

    child_df = load_df(child_path, pd.DataFrame(columns=["Patient_ID", "Mother_ID", "Father_ID"]))
    child_df.set_index('Patient_ID', inplace=True)
    child_df.to_csv(child_path)

    disease_df = load_df(disease_path, pd.DataFrame(columns=["Disease_ID", "Disease_name", "Diagnose_flag"]))
    disease_df.set_index('Disease_ID', inplace=True)
    disease_df.to_csv(disease_path)

    patient_disease_df = load_df(patient_disease_path, pd.DataFrame(columns=["PersonDisease_ID", "Patient_ID", "Disease_ID"]))
    patient_disease_df.set_index('PersonDisease_ID', inplace=True)
    patient_disease_df.to_csv(patient_disease_path)

    return patient_df, child_df, disease_df, patient_disease_df


patient_df, child_df, disease_df, patient_disease_df = load_from_path()


def add_person(patient_id, name, age, is_dead):
    patient_df.loc[patient_id] = [name, age, is_dead]
    patient_df.to_csv(patient_path)
    return patient_df


def add_child(patient_ID, mother_ID, father_ID):
    child_df.loc[patient_ID] = [mother_ID, father_ID]
    child_df.to_csv(child_path)
    return child_df

# Function to find relatives with a specific disease
def count_relatives_with_disease(patient_id, disease_id, child_df=child_df.reset_index(), patient_disease_df=patient_disease_df.reset_index()):
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



# def load_data(name, dataframe = None):
#     uploaded_file = st.file_uploader("Load {0} Data".format(name))
#     if uploaded_file is not None:
#         dataframe = pd.read_csv(uploaded_file)
#     return uploaded_file, dataframe


@st.dialog("Enter Person")
def person():
    nhs_number = st.text_input("PersonID (NHS Number)")
    name = st.text_input("Name")
    age = st.number_input("Age", 0, 200)
    is_dead = st.toggle('Is Dead')
    if st.button("Submit"):
        # st.session_state.person = {"patient_id": nhs_number, 'name': name, 'age': age, 'is_dead': is_dead}
        patient_df = add_person(nhs_number, name, age, is_dead)
        st.rerun()



def find_descendants(parent_id, df=child_df.reset_index()):
    # Initialize a set to store all descendants
    descendants = set()

    # Recursive function to find children and their descendants
    def find_children(parent_id):
        # Find children where the parent_id matches either Mother_ID or Father_ID
        children = df[(df['Mother_ID'] == parent_id) | (df['Father_ID'] == parent_id)]['Patient_ID']

        # For each child, add to descendants and find their children recursively
        for child in children:
            if child not in descendants:
                descendants.add(child)
                find_children(child)  # Recursively find this child's descendants

    # Start finding descendants from the given parent_id
    find_children(parent_id)

    return list(descendants)


def find_ancestors_by_level(df, patient_id, levels=2):
    """Find all ancestors (parents, grandparents) up to `levels`."""
    ancestors = set()

    def dfs_ancestors(pid, level):
        if level > levels:
            return
        # Find the row with the matching patient
        row = df[df["Patient_ID"] == pid]
        if not row.empty:
            mother = row.iloc[0]["Mother_ID"]
            father = row.iloc[0]["Father_ID"]
            if pd.notna(mother):
                ancestors.add(mother)
                dfs_ancestors(mother, level + 1)
            if pd.notna(father):
                ancestors.add(father)
                dfs_ancestors(father, level + 1)

    dfs_ancestors(patient_id, 1)
    return ancestors

def find_descendants_by_level(df, patient_id, levels=2):
    """Find all descendants (children, grandchildren) up to `levels`."""
    descendants = set()

    def dfs_descendants(pid, level):
        if level > levels:
            return
        # Find all rows where the patient is listed as a mother or father
        children = df[(df["Mother_ID"] == pid) | (df["Father_ID"] == pid)]["Patient_ID"]
        for child in children:
            descendants.add(child)
            dfs_descendants(child, level + 1)

    dfs_descendants(patient_id, 1)
    return descendants


def filter_family_tree(df, patient_id):
    """Filter the DataFrame to only include the patient, their ancestors (2 levels up), and descendants (2 levels down)."""
    # Find all relevant nodes
    ancestors = find_ancestors_by_level(df, patient_id)
    descendants = find_descendants_by_level(df, patient_id)

    # Include the patient itself
    relevant_nodes = ancestors | descendants | {patient_id}

    # Filter the DataFrame to only include the relevant nodes
    filtered_df = df[df["Patient_ID"].isin(relevant_nodes)]
    return filtered_df
    



def fetch_mother_father(patient_id):
    if child_df.empty:
        return None, None
    
    if patient_id not in child_df.index:
        return None, None
    
    mother_id = child_df.loc[patient_id, 'Mother_ID']
    father_id = child_df.loc[patient_id, 'Father_ID']
    return mother_id, father_id




@st.dialog("Enter Child")
def child():

    all_people = patient_df.index.to_list()

    patient_id = st.selectbox("Select Child", options=all_people, index=len(all_people)-1, format_func=lambda v: patient_df.loc[v, 'Name'])

    # Mother or Father can't be itself
    all_people = [a for a in all_people if a != patient_id]

    # Mother or Father can't be a descendant (prevent cycles)
    descendants = find_descendants(patient_id)
    all_people = list(set(all_people) - set(descendants))



   
    mother_index, father_index = None, None


    actual_mother, actual_father = fetch_mother_father(patient_id)


    if actual_mother is not None and pd.notna(actual_mother):
        mother_index = all_people.index(actual_mother)
        

    mother = st.selectbox("Select Mother", options=all_people, index=mother_index, format_func=lambda v: patient_df.loc[v, 'Name'])

    all_people = [a for a in all_people if a != mother]
    
    # if child_df is not None:
    #     actual_father = child_df.loc[child_df.Patient_ID == patient_id, 'Father_ID']
    #     if actual_father is not None:
    #         father_index = all_people.index(actual_father)


    if actual_father is not None and pd.notna(actual_mother):
        father_index = all_people.index(actual_father)

    father = st.selectbox("Select Father", options=all_people, index=father_index, format_func=lambda v: patient_df.loc[v, 'Name'])


    if st.button("Submit"):
        # st.session_state.person = {"patient_id": nhs_number, 'name': name, 'age': age, 'is_dead': is_dead}
        child_df = add_child(patient_id, mother, father)
        st.rerun()


def add_disease(disease_ID,disease_name):
    disease_df.loc[disease_ID] = [disease_name]
    disease_df.to_csv(disease_path)
    return disease_df



@st.dialog("Enter Disease")
def disease():
    disease_ID = st.text_input("Enter disease code")
    ddisease_name = st.text_input("Enter disease name")
    if st.button("Submit"):
        disease_df = add_disease(disease_ID,ddisease_name)
        st.rerun()



def fetch_diseases(patient_id):
    diseases = patient_disease_df.loc[patient_disease_df.Patient_ID==patient_id]
    if diseases.empty:
        return []
    return diseases.loc[:, 'Disease_ID'].tolist()


def drop_patient_in_patient_disease(patient_id, patient_disease_df=patient_disease_df):
    patient_disease_df = patient_disease_df.loc[ ~(patient_disease_df.Patient_ID == patient_id) ]
    patient_disease_df.to_csv(patient_disease_path)
    return patient_disease_df

def add_patient_disease(patient_disease_tuples, patient_disease_df):
    max_id = 0 if patient_disease_df.empty else patient_disease_df.index.max() + 1
    for patient_ID, disease_ID in patient_disease_tuples:
        patient_disease_df.loc[max_id] = [patient_ID, disease_ID]
        max_id += 1
    patient_disease_df.to_csv(patient_disease_path)
    return patient_disease_df


# PersonDisease_ID,Patient_ID,Disease_ID
@st.dialog("Assign Disease")
def patient_disease():
    all_people = patient_df.index.to_list()
    patient_id = st.selectbox("Select Patient", options=all_people, index=len(all_people)-1, format_func=lambda v: patient_df.loc[v, 'Name'], key='select_patient_filter')

    all_diseases = disease_df.index.to_list()
    found_diseases = fetch_diseases(patient_id)
    disease_ids = st.multiselect("Select Diseases", options=all_diseases, default=found_diseases, format_func=lambda v: disease_df.loc[v, 'Disease_name'], key='select_disease_filter')


    if st.button("Submit"):
        patient_disease_df = drop_patient_in_patient_disease(patient_id)
        n = len(disease_ids)
        if n > 0:
            tups = list(zip([patient_id for i in range(n)], disease_ids))
            print('\n\n\n\n', tups, '\n\n\n')
            patient_disease_df = add_patient_disease(tups, patient_disease_df)
        # disease_df = add_disease(disease_ID,ddisease_name)
        st.rerun()




with st.sidebar:
    manual_entry = st.toggle("Manually Enter Data")
    if manual_entry:
        if st.button("Person"):
            person()
        if st.button('Child'):
            child()
        if st.button('Disease'):
            disease()
        if st.button('Assign diseases'):
            patient_disease()
    





def convert_child_df_to_edge_list(child_df):
    # Convert the Child_df  to a list of dictionaries
    linking_data = child_df.to_dict(orient='records')

    print(linking_data)
    rows = []
    # Iterate through each dictionary in the linking_data list
    for entry in linking_data:
        patient_id = entry["Patient_ID"]
        
        # Add a row for the mother relationship
        if "Mother_ID" in entry:
            mid = entry["Mother_ID"]
            if pd.notna(mid):
                rows.append({
                    "source": entry["Mother_ID"],
                    "target": patient_id,
                    "relationship": "mother"
                })
        
        # Add a row for the father relationship
        if "Father_ID" in entry:
            fid = entry["Father_ID"]
            if pd.notna(fid):
                rows.append({
                    "source": entry["Father_ID"],
                    "target": patient_id,
                    "relationship": "father"
                })
    
    # Create a DataFrame from the rows list
    linking_df = pd.DataFrame(rows)
    
    return linking_df









# Validating data:


# Step 1: Create the networkx graph
def generate_family_tree(selected_disease=None, selected_patient=None):

    family_tree = child_df if selected_patient is None else filter_family_tree(child_df.reset_index(), selected_patient)
    linking_df = convert_child_df_to_edge_list(family_tree.reset_index())
    G = nx.from_pandas_edgelist(linking_df, 'source', 'target', ['relationship'], create_using=nx.DiGraph())

    
    # T = nx.balanced_tree(2, 3)

    pos = graphviz_layout(G, prog="dot")

    # pos = nx.spring_layout(G)

    # Create a binary tree with 7 nodes
    # T = nx.balanced_tree(r=2, h=2)  # r=2 (binary), h=2 (height)

    # Use graphviz_layout to arrange the tree
    # pos_t = nx.nx_agraph.graphviz_layout(T, prog="dot")   

    to_name = lambda id: patient_df.loc[id, 'Name'] if pd.notna(id) else None




    find_mother = lambda id: to_name(child_df.loc[id, 'Mother_ID']) if id in child_df.index  else None

    find_father= lambda id: to_name(child_df.loc[id, 'Father_ID']) if id in child_df.index else None

    to_disease_name = lambda d_id: disease_df.loc[d_id, 'Disease_name'] if pd.notna(d_id) else None

    is_carrier = lambda id, disease_id: len(patient_disease_df.loc[(patient_disease_df.Patient_ID==id) & (patient_disease_df.Disease_ID==disease_id)]) > 0

    if selected_disease is not None:
        selected_disease_name = to_disease_name(selected_disease)


    # Add attributes to nodes
    for n in G.nodes():
        G.nodes[n]['id'] = n
        G.nodes[n]['name'] = to_name(n)
        G.nodes[n]['mother'] = find_mother(n)
        G.nodes[n]['father'] = find_father(n)
        if selected_disease is not None:
            G.nodes[n][selected_disease_name] = is_carrier(n, selected_disease)
        G.nodes[n]['selected'] = 0.5 + 0.5*(1 if n==selected_patient else 0)
    
    relationship_colors = {
    'mother': 'pink',
    'father': 'blue'
    }

    for e in G.edges():
        # G.edges[e[0], e[1]]['n'] = str(e[0]) + " to " + str(e[1])
        rel = G.edges[e[0], e[1]]['relationship']
        G.edges[e[0], e[1]]['color'] = relationship_colors[rel]
        if selected_patient is not None and (e[0]==selected_patient or e[1]==selected_patient):
            G.edges[e[0], e[1]]['selected'] = True

        

    if selected_disease is not None:


    
        chart = nxa.draw_networkx(
            G=G,
            pos=pos,
            width=10,
            node_color=selected_disease_name,
            cmap='set2',
            node_tooltip=['id', 'name', 'father', 'mother', selected_disease_name],
            edge_color='selected',

            node_size=600,

        )
    else:
        chart = nxa.draw_networkx(
            G=G,
            pos=pos,
            width=10,
            node_color='green',
            node_tooltip=['id', 'name', 'father', 'mother'],
            edge_color='cyan',
            node_size=600

        )

        # chart_edges = chart.layer[0]
        # chart_nodes = chart.layer[1]
        # pick_name = alt.selection_point(fields=['id'], empty=True)
        # chart_nodes = chart_nodes.encode(
        #     opacity = alt.condition(pick_name, alt.value(1.0), alt.value(1.0))
        # ).add_params(
        #     pick_name
        # )
        # return (chart_nodes + chart_edges)

    


    

    

    return chart


    




raw, tree, test = st.tabs(["Raw", "Tree Viz", 'test3'])


with raw:
    st.header('Patient Data')
    st.write(patient_df)
    st.header('Parents Data')
    st.write(child_df)
    st.header('Disease Data')
    st.write(disease_df)
    st.header('Patient Disease Data')
    st.write(patient_disease_df)
    
with tree:
    with st.expander('Hereditary Risk'):
        filter_on = st.toggle('Determine the risk')
        all_diseases = disease_df.index.to_list()
        selected_disease = st.selectbox('Select Disease', all_diseases, format_func=lambda v: disease_df.loc[v, 'Disease_name'])
        all_people = patient_df.index.to_list()
        selected_patient = st.selectbox("Select Patient", options=all_people, index=len(all_people)-1, format_func=lambda v: patient_df.loc[v, 'Name'])
        if not filter_on:
            selected_disease, selected_patient = None, None
        else:
            result, verdict = count_relatives_with_disease(selected_patient, selected_disease)
            
            st.write(result.to_frame('Answer'))

            st.markdown('Verdict: ' + verdict)

        




    # Generate and display the family tree
    # family_tree = generate_family_tree()
    family_tree_chart = generate_family_tree(selected_disease, selected_patient)
    st.altair_chart(family_tree_chart, use_container_width=True)






import random
import matplotlib.pyplot as plt



with test:
    # Initialize Streamlit state for the selected patient
    if "selected_node" not in st.session_state:
        st.session_state.selected_node = None

    # Sample family graph creation
    G = nx.Graph()

    # Define relationships (parents -> children, 'marriage' -> represents "-")
    relationships = {
        (1, 2): [3, 4, 5],  # 1 and 2 are parents, 3, 4, 5 are children
        (3, 6): [7],         # 3 and 6 are parents, 7 is their child
        (4, 8): [],          # 4 and 8 are parents, no children
        (5,): []             # 5 has no children
    }

    # Add edges to the graph based on relationships
    for parents, children in relationships.items():
        for child in children:
            G.add_edge(parents[0], child)  # Parent to child

    # Add 'marriage' connections (e.g., between 1 and 2)
    for parents in relationships:
        if len(parents) == 2:  # It's a pair of parents
            G.add_edge(parents[0], parents[1])  # Marriage connection

    # Generate random patient data
    random.seed(42)  # For reproducibility
    patient_names = [f"Patient_{i}" for i in range(1, 9)]
    disease_presence = [random.choice([True, False]) for _ in range(8)]

    # Store node information in a DataFrame
    node_data = pd.DataFrame({
        'node': list(G.nodes),
        'patient_name': patient_names,
        'has_disease': disease_presence
    })

    # Layout for graph drawing
    pos = nx.spring_layout(G)

    def draw_graph(selected_node=None):
        """Draw the graph with highlights based on the selected node."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Draw the entire graph with default node colors
        nx.draw(G, pos, with_labels=True, node_color='lightgray', node_size=500, font_size=10, ax=ax)

        # Highlight nodes with disease
        diseased_nodes = node_data[node_data['has_disease'] == True]['node'].tolist()
        nx.draw_networkx_nodes(G, pos, nodelist=diseased_nodes, node_color='black', node_size=700, ax=ax)

        if selected_node is not None:
            # First generation (direct neighbors)
            gen_1_nodes = set(G.neighbors(selected_node))

            # Second generation (neighbors of neighbors)
            gen_2_nodes = set()
            for node in gen_1_nodes:
                gen_2_nodes.update(G.neighbors(node))
            gen_2_nodes.difference_update({selected_node}, gen_1_nodes)

            # Draw nodes with generation-based highlighting
            nx.draw_networkx_nodes(G, pos, nodelist=[selected_node], node_color='yellow', node_size=700, ax=ax)
            nx.draw_networkx_nodes(G, pos, nodelist=gen_1_nodes, node_color='none', edgecolors='blue', linewidths=2, ax=ax)
            nx.draw_networkx_nodes(G, pos, nodelist=gen_2_nodes, node_color='none', edgecolors='green', linewidths=2, ax=ax)

            # Add legend with patient details
            patient_name = node_data[node_data['node'] == selected_node]['patient_name'].values[0]
            has_disease = node_data[node_data['node'] == selected_node]['has_disease'].values[0]
            legend_text = f"Selected Patient: {patient_name}\nDisease: {'Yes' if has_disease else 'No'}\n"
            ax.text(0.95, 0.05, legend_text, transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
                    horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))

        st.pyplot(fig)  # Display the graph in Streamlit

    # Streamlit UI
    st.title("Interactive Family Tree Viewer")

    # Dropdown to select a patient node
    selected_patient = st.selectbox("Select a Patient:", options=list(G.nodes), format_func=lambda x: f"Patient {x}")

    # Update the selected node in the session state
    if selected_patient:
        st.session_state.selected_node = selected_patient

    # Draw the graph with the selected node highlighted
    draw_graph(st.session_state.selected_node)



