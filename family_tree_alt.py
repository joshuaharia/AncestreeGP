import streamlit as st
import networkx as nx
import nx_altair as nxa
import altair as alt
import pandas as pd
from networkx.drawing.nx_pydot import graphviz_layout


from Ancestory.model import AncestoryModel

@st.cache_resource
def set_up_model():
    print('\n\n\n\n\n\n', 'SETTING UP NEW MODEL', '\n\n\n\n\n')
    return AncestoryModel('Data/patients.csv', 'Data/child.csv', 'Data/disease.csv', 'Data/patient_disease.csv')

model = set_up_model()


@st.dialog("Enter Person")
def person():
    nhs_number = st.text_input("PersonID (NHS Number)")
    name = st.text_input("Name")
    age = st.number_input("Age", 0, 200)
    is_dead = st.toggle('Is Dead')
    if st.button("Submit"):
        model.update_patient(nhs_number, name, age, is_dead)
        st.rerun()


@st.dialog("Enter Child")
def child():
    all_people = model.people()

    patient_id = st.selectbox("Select Child", options=all_people, index=len(all_people)-1, format_func=model.name)

    # Mother or Father can't be itself
    all_people = [a for a in all_people if a != patient_id]

    # Mother or Father can't be a descendant (prevent cycles)
    descendants = model.find_descendants(patient_id)
    all_people = list(set(all_people) - set(descendants))
   
    mother_index, father_index = None, None
    actual_mother, actual_father = model.fetch_mother(patient_id), model.fetch_father(patient_id)

    if actual_mother is not None and pd.notna(actual_mother):
        mother_index = all_people.index(actual_mother)
        
    mother = st.selectbox("Select Mother", options=all_people, index=mother_index, format_func=model.name)

    all_people = [a for a in all_people if a != mother]
    
    if actual_father is not None and pd.notna(actual_father):
        father_index = all_people.index(actual_father)

    father = st.selectbox("Select Father", options=all_people, index=father_index, format_func=model.name)

    if st.button("Submit"):
        model.update_child(patient_id, mother, father)
        st.rerun()

@st.dialog("Enter Disease")
def disease():
    disease_ID = st.text_input("Enter disease code")
    ddisease_name = st.text_input("Enter disease name")
    if st.button("Submit"):
        model.update_disease(disease_ID,ddisease_name)
        st.rerun()

@st.dialog("Assign Disease")
def patient_disease():
    all_people = model.people()
    patient_id = st.selectbox("Select Patient", options=all_people, index=len(all_people)-1, format_func=lambda v: patient_df.loc[v, 'Name'], key='select_patient_filter')

    all_diseases = model.diseases(None)
    found_diseases = model.diseases(patient_id)
    disease_ids = st.multiselect("Select Diseases", options=all_diseases, default=found_diseases, format_func=lambda v: disease_df.loc[v, 'Disease_name'], key='select_disease_filter')

    if st.button("Submit"):
        model.update_diseases_for_patient(patient_id, disease_ids)
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


def generate_family_tree(selected_disease=None, selected_patient=None):

    if selected_patient is None:
        linking_df = model.get_edge_list()
    else:
        linking_df = model.filter_family_tree(selected_patient, True)

    if linking_df.empty:
        st.error('No tree can be displayed for {0}'.format(model.name(selected_patient)), icon="🚨")
        return None


    G = nx.from_pandas_edgelist(linking_df, 'source', 'target', ['relationship'], create_using=nx.DiGraph())

    pos = graphviz_layout(G, prog="dot")

    if selected_disease is not None:
        selected_disease_name = model.disease_name(selected_disease)

    # Add attributes to nodes
    for n in G.nodes():
        G.nodes[n]['id'] = n
        G.nodes[n]['name'] = model.name(n)
        G.nodes[n]['mother'] = model.name(model.fetch_mother(n))
        G.nodes[n]['father'] = model.name(model.fetch_father(n))
        G.nodes[n]['diseases'] = ','.join(map(model.disease_name, model.diseases(n)))
        if selected_disease is not None:
            G.nodes[n][selected_disease_name] = selected_disease in model.diseases(n)
        G.nodes[n]['selected'] = n==selected_patient
    
    for e in G.edges():
        if selected_patient is not None and (e[0]==selected_patient or e[1]==selected_patient):
            G.edges[e[0], e[1]]['selected'] = True

    if selected_disease is not None:
        chart = nxa.draw_networkx(
            G=G,
            pos=pos,
            width=5,
            node_color=selected_disease_name,
            cmap='set2',
            node_tooltip=['id', 'name', 'father', 'mother', selected_disease_name],
            edge_color='selected',
            node_size=600)
        
        chart_edges = chart.layer[0]
        chart_b = chart.layer[1]
        chart_nodes = chart.layer[2]


        # st.write('Layer' + str(len(chart.layer)))

        chart_nodes = chart_nodes.encode(
            opacity = alt.when(alt.datum.id == selected_patient).then(alt.value(1)).otherwise(alt.value(0.9)),
            stroke=alt.when(alt.datum.id == selected_patient).then(alt.value('white')).otherwise(alt.value('black')),
            strokeWidth=alt.when(alt.datum.id == selected_patient).then(alt.value(4)).otherwise(alt.value(2)),
        )

        chart_edges = chart_edges.encode(
            opacity = alt.value(0.6)
        )

        chart = (chart_edges + chart_nodes).interactive()


    else:
        chart = nxa.draw_networkx(
            G=G,
            pos=pos,
            width=5,
            node_color='green',
            node_tooltip=['id', 'name', 'father', 'mother', 'diseases'],
            edge_color='cyan',
            node_size=600)
        
        chart_edges = chart.layer[0]
        chart_b = chart.layer[1]
        chart_nodes = chart.layer[2]


        # st.write('Layer' + str(len(chart.layer)))

        chart_nodes = chart_nodes.encode(
            opacity = alt.value(0.9)
        )

        chart_edges = chart_edges.encode(
            opacity = alt.value(0.6)
        )

        chart = (chart_edges + chart_nodes).interactive()
        

    return chart



raw, tree, test = st.tabs(["Raw", "Tree Viz", 'test3'])


with raw:
    st.header('Patient Data')
    st.write(model.fetch_data('patient'))
    st.header('Parents Data')
    st.write(model.fetch_data('child'))
    st.header('Disease Data')
    st.write(model.fetch_data('disease'))
    st.header('Patient Disease Data')
    st.write(model.fetch_data('patient_disease'))
    
with tree:
    with st.expander('Hereditary Risk'):
        filter_on = st.toggle('Determine the risk')
        all_diseases = model.diseases(None)
        selected_disease = st.selectbox('Select Disease', all_diseases, format_func=model.disease_name)
        all_people = model.people()
        selected_patient = st.selectbox("Select Patient", options=all_people, index=len(all_people)-1, format_func=model.name)
        if not filter_on:
            selected_disease, selected_patient = None, None
        else:
            result, verdict = model.count_relatives_with_disease(selected_patient, selected_disease)
            st.write(result.to_frame('Answer'))
            st.markdown('Suggestion: ' + verdict)

    # Generate and display the family tree
    # family_tree = generate_family_tree()

    family_tree_chart = generate_family_tree(selected_disease, selected_patient)
    if family_tree_chart is not None:
        st.altair_chart(family_tree_chart, use_container_width=True)



#***********************************************************
#                                                          *
#                                                          *
#               ALTERNATIVE SOLUTION                       *
#                                                          *
#                                                          *
# **********************************************************

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



