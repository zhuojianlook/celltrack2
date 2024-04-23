import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import numpy as np
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import MaxNLocator

st.set_page_config(layout="wide")

# Helper functions used in the main function
if 'graph' not in st.session_state:
    st.session_state['graph'] = nx.DiGraph()
if 'depth_counters' not in st.session_state:
    st.session_state['depth_counters'] = defaultdict(lambda: defaultdict(int))

def authenticate_gspread(json_key):
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(json_key, scope)
    gc = gspread.authorize(credentials)
    return gc

def calculate_pd(cells_start, cells_end):
    if cells_start > 0 and cells_end > 0:
        return np.log10(cells_end / cells_start) / np.log10(2)
    else:
        return np.nan

def calculate_cpd(pd_values):
    return np.cumsum(pd_values)

def extract_cell_lines(nodes):
    cell_lines = set()
    for node in nodes:
        parts = node.split('P')
        if parts:
            cell_lines.add(parts[0])
    return list(cell_lines)

def load_data(sheet_url, gc):
    worksheet = gc.open_by_url(sheet_url).sheet1
    data = worksheet.get_all_records()
    df = pd.DataFrame(data)
    expected_headers = ['Node', 'Parent', 'Date', 'Vessel Type', 'Cells Start', 'Cells End', 'Notes']
    for header in expected_headers:
        if header not in df.columns:
            df[header] = pd.to_numeric([], errors='coerce') if header in ['Cells Start', 'Cells End'] else ''
    df['Cells Start'] = pd.to_numeric(df['Cells Start'], errors='coerce')
    df['Cells End'] = pd.to_numeric(df['Cells End'], errors='coerce')
    return df

def handle_cell_line_selection(unique_cell_lines):
    if 'selected_cell_lines' not in st.session_state or not st.session_state.selected_cell_lines:
        st.session_state.selected_cell_lines = st.multiselect('Select cell lines:', unique_cell_lines, key='cell_line_selector')
    else:
        st.session_state.selected_cell_lines = st.multiselect('Select cell lines:', unique_cell_lines, default=st.session_state.selected_cell_lines)

def save_data(df, sheet_url, gc):
    worksheet = gc.open_by_url(sheet_url).sheet1
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
    df.fillna('', inplace=True)
    df.replace([np.inf, -np.inf], '', inplace=True)
    current_data = worksheet.get_all_values()
    if not current_data or current_data == [[]]:
        worksheet.update([df.columns.values.tolist()] + df.values.tolist())
    else:
        range_to_clear = f"A2:Z{len(current_data)}"
        worksheet.batch_clear([range_to_clear])
        worksheet.append_rows(df.values.tolist(), value_input_option='USER_ENTERED')

def add_nodes(parent_node, num_children, creation_datetime, vessel_types, num_cells_start, num_cells_end_parent, notes):
    if isinstance(creation_datetime, str):
        creation_datetime = datetime.strptime(creation_datetime, "%Y-%m-%d %H:%M:%S")
    if parent_node not in st.session_state['graph']:
        st.session_state['graph'].add_node(parent_node, date=creation_datetime, depth=-1)
    st.session_state['graph'].nodes[parent_node]['num_cells_end'] = num_cells_end_parent
    base_name = parent_node.split('P')[0]
    current_depth = st.session_state['graph'].nodes[parent_node]['depth']
    next_depth = current_depth + 1
    for i in range(num_children):
        child_index = st.session_state['depth_counters'][base_name][next_depth]
        child_node = f"{base_name}P{next_depth}.{child_index}"
        st.session_state['graph'].add_node(child_node, date=creation_datetime, depth=next_depth, vessel_type=vessel_types[i], num_cells_start=num_cells_start[i], notes=notes[i])
        st.session_state['graph'].add_edge(parent_node, child_node)
        st.session_state['depth_counters'][base_name][next_depth] += 1

def draw_graph():
    pos = nx.multipartite_layout(st.session_state['graph'], subset_key="depth")
    plt.figure(figsize=(20, 15))
    colors = {'T75': '#fbb4ae', 'T25': '#b3cde3', 'T125': '#decbe4', '12 well plate': '#ccebc5', '6 well plate': '#fed9a6', 'Cryovial': '#fddaec', 'Unknown': '#cccccc'}
    labels = {node: f"{node}\nDate: {data['date']}\nVessel: {data.get('vessel_type', 'Unknown')}\nCells start: {data.get('num_cells_start', 'N/A')}\nCells end: {data.get('num_cells_end', 'N/A')}\nNotes: {data.get('notes', '')}" for node, data in st.session_state['graph'].nodes(data=True)}
    node_colors = [colors.get(data.get('vessel_type', 'Unknown'), '#cccccc') for _, data in st.session_state['graph'].nodes(data=True)]
    nx.draw(st.session_state['graph'], pos, labels=labels, with_labels=True, node_size=7000, node_color=node_colors, font_size=9, font_color="black", arrowstyle='-|>', arrowsize=10)
    plt.savefig("graph.png")
    plt.close()
    st.image("graph.png", use_column_width=True)

def reconstruct_graph(df):
    st.session_state['graph'].clear()
    st.session_state['depth_counters'].clear()
    for _, row in df.iterrows():
        node = row['Node']
        depth = -1 if 'P' not in node else int(node.split('P')[1].split('.')[0])
        st.session_state['graph'].add_node(node, date=row.get('Date', 'N/A'), vessel_type=row.get('Vessel Type', 'Unknown'), num_cells_start=row.get('Cells Start', 'N/A'), num_cells_end=row.get('Cells End', 'N/A'), notes=row.get('Notes', ''), depth=depth)
        if depth >= 0:
            base_name = node.split('P')[0]
            st.session_state['depth_counters'][base_name][depth] = max(st.session_state['depth_counters'][base_name][depth], int(node.split('.')[1]))
    for _, row in df.iterrows():
        child_node = row['Node']
        parent_node = row['Parent']
        if pd.notnull(parent_node) and parent_node in st.session_state['graph']:
            st.session_state['graph'].add_edge(parent_node, child_node)
    for node in st.session_state['graph'].nodes:
        if 'P' not in node and st.session_state['graph'].out_degree(node) == 0:
            st.session_state['graph'].nodes[node]['depth'] = 0

def plot_graphs(cpd_data, dt_data):
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    for cell_line, values in cpd_data.items():
        valid_values = [value for value in values if isinstance(value, (int, float)) and not np.isnan(value)]
        if len(valid_values) > 0:
            x = np.arange(len(valid_values))
            if len(valid_values) > 3:
                x_new = np.linspace(x.min(), x.max(), 300)
                spl = make_interp_spline(x, valid_values, k=3)
                y_smooth = spl(x_new)
                plt.plot(x_new, y_smooth, label=f'{cell_line}')
            else:
                plt.plot(x, valid_values, label=f'{cell_line}', linestyle='--')
            plt.scatter(x, valid_values, marker='o')
    ax1 = plt.gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title('Cumulative Population Doublings (cPD) vs. Passage Number')
    plt.xlabel('Passage Number')
    plt.ylabel('cPD')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    for cell_line, values in dt_data.items():
        valid_values = [value for value in values if isinstance(value, (int, float)) and not np.isnan(value)]
        if len(valid_values) > 0:
            x = np.arange(len(valid_values))
            if len(valid_values) > 3:
                try:
                    x_new = np.linspace(x.min(), x.max(), 300)
                    spl = make_interp_spline(x, valid_values, k=3)
                    y_smooth = spl(x_new)
                    plt.plot(x_new, y_smooth, label=f'{cell_line}')
                except Exception as e:
                    print(f"Interpolation error for {cell_line}: {e}")
                    plt.plot(x, valid_values, label=f'{cell_line}', linestyle='--')
            else:
                plt.plot(x, valid_values, label=f'{cell_line}', linestyle='--')
            plt.scatter(x, valid_values, marker='o')
    ax2 = plt.gca()
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title('Doubling Time (DT) vs. Passage Number')
    plt.xlabel('Passage Number')
    plt.ylabel('DT (hours/PD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt.gcf())

def main():
    st.title('Cell Culture Tracking Program')
    default_base_node = ''
    if 'reset' in st.session_state and st.session_state.reset:
        default_base_node = ''
    else:
        default_base_node = st.session_state.get('new_base_node', '')

    st.sidebar.title("Instructions")
    st.sidebar.markdown("""                 
        ### How to Use This Application
        To effectively use this application, follow these steps:
        1. Upload your Google service account JSON key.
        2. Enter the URL of your Google Sheet that is shared with the service account.
        3. Use the controls to add culture steps and define passage parameters.
        4. Click 'Add to Graph' to create and visualize the graph.
        5. Use 'Save Data to Sheet' to save your graph data to Google Sheets.
        6. Use 'Load Data from Sheet' to load and reconstruct the graph from Google Sheets.
        7. Press 'Refresh' if you need to refresh the session state, e.g., after adding a cell line.

        ### Best Practices for Cell Culture Calculations and Methodologies
        #### Understanding Population Doubling (PD) and Cumulative Population Doubling (cPD)
        - **Population Doubling (PD)** is used to measure the growth rate of cells in culture, reflecting the number of times the cell population has doubled during a passage.
        - **Cumulative Population Doubling (cPD)** represents the total number of times the cell population has doubled since the onset of the culture. It is a useful metric for comparing the biological age of cells from different cell lines.

        #### Calculating Population Doubling (PD)
        To calculate the population doubling during a specific passage:
        1. **Formula**:
        $PD = \frac{\log\left(\frac{\text{Number of cells at harvest}}{\text{Number of cells seeded}}\right)}{\log(2)}$
        - Convert the ratio of harvested cells to seeded cells into a logarithmic scale using log base 10, and then divide by log(2) to convert it into base 2.
        - This calculation gives the number of times the cell population has doubled during the passage.

        #### Calculating Doubling Time (DT)
        Doubling time reflects how quickly cells double in number during a culture period.
        1. **Formula**:
        $DT = \frac{\text{Time in culture (e.g., days or hours)}}{PD}$
        - Divide the total culture duration by the population doubling to find how often the cells doubled during that time.

        #### Tracking Cumulative Population Doubling (cPD)
        - To obtain the cumulative population doubling, add the PD of each passage sequentially.
        - Example progression:
        - Passage 1 (P1): PD = 3.5
        - Passage 2 (P2): PD = 3.0 (Cumulative PD = 6.5)
        - Passage 3 (P3): PD = 2.5 (Cumulative PD = 9.0)
        - This series provides a running total of cell doublings, useful for assessing cell line vitality and comparability across different lines.

        #### Plotting Growth Curves
        - Plot cPD against the passage number to visualize the growth or expansion curve of the cell line.
        - This curve helps compare the growth rates and senescence stages across different cell lines.

        #### Practical Tips for Cell Culture
        - Always maintain the recommended minimum seeding density specific to each cell line.
        - While not strictly required for calculations, standardize seeding density for experiments to ensure reproducibility
        - Harvest cells based on fixed time points or confluence levels (e.g., 80-90% confluence).
        - Document all relevant parameters:
        - Days in culture.
        - Seeding and harvest cell count numbers (for PD and DT calculations).
        - Maintain a consistent type/number of flasks and media volumes used (important for scaling and reproducibility). e.g. 0.1 ml of media per cm²              

        ### Setting Up Prerequisites
        To set up prerequisites for using this application, follow these steps:
        1. **Create a Google service account JSON Key**:
            - **Step 1**: Create a [Google Cloud Project](https://console.cloud.google.com/welcome?project=cell-culture-tracking).
            - Go to the Google Cloud Console.
            - Sign in with your Google account if you haven't already done so.
            - Click on the project dropdown at the top of the page and then click on "New Project".
            - Enter a project name and select a billing account if necessary.
            - Click on "Create".
            - **Step 2**: Enable Google Sheets API.
            - Make sure the new project is selected.
            - Navigate to "APIs & Services > Dashboard".
            - Click on "+ ENABLE APIS AND SERVICES".
            - Search for "Google Sheets API" and click on it.
            - Click on "Enable".
            - **Step 3**: Create a Service Account.
            - Go to "IAM & Admin > Service accounts".
            - Click on "Create Service Account".
            - Enter a name and description for the service account.
            - Click on "Create and Continue".
            - You can skip granting this service account access to the project and click "Continue".
            - Click "Done" to finish creating the service account.
            - **Step 4**: Create the JSON Key.
            - Find the service account you just created in the list and click on it.
            - Go to the "Keys" tab.
            - Click on "Add Key" and choose "JSON" from the dropdown menu.
            - Your browser will download a JSON file containing the private key. Keep it safe.
        2. Create a Google sheet with your normal Google account and share the Google sheet with the service account email with 'write' privileges. The service account email can be found in the JSON file and should look like 'example@example.iam.gserviceaccount.com'.
        """)

    uploaded_file = st.file_uploader("Upload Google service account JSON key", type="json")
    sheet_url = st.text_input("Enter the URL of your Google Sheet")
    gc = None
    if uploaded_file and sheet_url:
        json_key = json.load(uploaded_file)
        gc = authenticate_gspread(json_key)

    num_children = st.number_input('Enter Number of Child Vessels', min_value=0, step=1, format="%d")
    vessel_options = ['T75', 'T25', 'T125', '12 well plate', '6 well plate', 'Cryovial']
    vessel_selections = [st.selectbox(f'Vessel type for child {i+1}:', vessel_options, key=f'vessel_{i}') for i in range(num_children)]
    num_cells_start = [st.number_input(f'Total start cell number seeded for child {i+1}:', min_value=0, format="%d", key=f'cells_start_{i}') for i in range(num_children)]
    notes = [st.text_input(f'Notes for child {i+1}:', key=f'notes_{i}') for i in range(num_children)]
    num_cells_end_parent = st.number_input('Total end cell number for Parent vessel', min_value=0, format="%d", key='cells_end_parent')
    creation_date = st.date_input('Select Date of Passage', value=datetime.today())
    creation_time = st.time_input('Select Time of Passage', value=datetime.now().time())
    full_creation_datetime = datetime.combine(creation_date, creation_time)

    existing_nodes = [node for node in st.session_state['graph']]
    base_node_selection = st.selectbox('Select an existing vessel', [""] + existing_nodes, index=0, key="base_node_selection")
    new_base_node = st.text_input('Or enter a new lineage name', value=default_base_node, key="new_base_node")

    if st.button('Add Entry'):
        base_node = new_base_node if new_base_node else base_node_selection
        if base_node:
            add_nodes(base_node, num_children, full_creation_datetime, vessel_selections, num_cells_start, num_cells_end_parent, notes)

    if st.button('Refresh'):
        st.session_state.reset = True
        st.rerun()

    if 'reset' in st.session_state and st.session_state.reset:
        st.session_state.reset = False

    if gc and sheet_url and st.button('Save Data to Sheet'):
        nodes_data = [
            {
                'Node': node,
                'Parent': list(st.session_state['graph'].predecessors(node))[0] if list(st.session_state['graph'].predecessors(node)) else None,
                'Date': data['date'],
                'Vessel Type': data.get('vessel_type', 'Unknown'),
                'Cells Start': data.get('num_cells_start', 'N/A'),
                'Cells End': data.get('num_cells_end', 'N/A'),
                'Notes': data.get('notes', '')
            } for node, data in st.session_state['graph'].nodes(data=True)
        ]
        df_to_save = pd.DataFrame(nodes_data)
        save_data(df_to_save, sheet_url, gc)
        st.success("Graph data saved successfully to Google Sheets.")

    if gc and sheet_url and st.button('Load Data from Sheet'):
        df = load_data(sheet_url, gc)
        reconstruct_graph(df)
        st.success("Data loaded and graph reconstructed successfully from Google Sheets.")
        draw_graph()

    if gc and sheet_url:
        df = load_data(sheet_url, gc)
        unique_cell_lines = extract_cell_lines(df['Node'])
        selected_cell_lines = st.multiselect('Select cell lines:', unique_cell_lines, key='cell_line_selection')

        if 'selected_flasks' not in st.session_state:
            st.session_state['selected_flasks'] = {}

        for cell_line in selected_cell_lines:
            cell_line_data = df[df['Node'].str.contains(f'^{cell_line}P')]
            passage_dict = defaultdict(list)
            for node in cell_line_data['Node']:
                parts = node.split('P')
                if len(parts) > 1 and '.' in parts[1]:
                    passage = int(parts[1].split('.')[0])
                    passage_dict[passage].append(node)

            sorted_passages = sorted(passage_dict.keys())
            if cell_line not in st.session_state['selected_flasks']:
                st.session_state['selected_flasks'][cell_line] = {}

            for passage in sorted_passages:
                flask_options = passage_dict[passage]
                selected_flask_key = f"{cell_line}_passage_{passage}"
                if selected_flask_key not in st.session_state['selected_flasks'][cell_line]:
                    st.session_state['selected_flasks'][cell_line][selected_flask_key] = flask_options[0] if flask_options else None
                current_selection = st.session_state['selected_flasks'][cell_line][selected_flask_key]
                st.session_state['selected_flasks'][cell_line][selected_flask_key] = st.selectbox(
                    f'Select flask for {cell_line} passage {passage}:',
                    flask_options,
                    key=selected_flask_key,
                    index=flask_options.index(current_selection) if current_selection in flask_options else 0
                )

        if st.button('Calculate PD, DT, and Generate Graphs', key='calculate_graphs_button'):
            pd_data = {}
            cpd_data = {}
            time_in_culture_data = {}
            dt_data = {}
            max_passage = 0

            for cell_line in selected_cell_lines:
                pd_values = []
                time_in_culture_values = []
                dt_values = []
                
                for _, flask in st.session_state['selected_flasks'][cell_line].items():
                    flask_data = df[df['Node'] == flask].iloc[0]
                    cells_start = flask_data['Cells Start']
                    cells_end = flask_data['Cells End']
                    pd_value = calculate_pd(cells_start, cells_end)
                    pd_values.append(pd_value)
                    
                    if flask in st.session_state['graph']:
                        child_nodes = list(st.session_state['graph'].successors(flask))
                        if child_nodes:
                            child_node = child_nodes[0]
                            parent_datetime = pd.to_datetime(st.session_state['graph'].nodes[flask]['date'])
                            child_datetime = pd.to_datetime(st.session_state['graph'].nodes[child_node]['date'])
                            time_diff = child_datetime - parent_datetime
                            time_in_culture_hours = time_diff.total_seconds() / 3600
                            time_in_culture_values.append(time_in_culture_hours)
                            if pd_value != 0:
                                dt_values.append(time_in_culture_hours / pd_value)
                            else:
                                dt_values.append(None)
                        else:
                            time_in_culture_values.append(None)
                            dt_values.append(None)
                
                pd_data[cell_line] = pd_values
                cpd_data[cell_line] = np.cumsum(pd_values)
                time_in_culture_data[cell_line] = time_in_culture_values
                dt_data[cell_line] = dt_values
                max_passage = max(max_passage, len(pd_values))

            max_passages = [len(pd_data[cl]) for cl in selected_cell_lines]
            all_pd = [pd for cl in selected_cell_lines for pd in pd_data[cl]]
            all_times = [time for cl in selected_cell_lines for time in time_in_culture_data[cl]]
            all_dts = [dt for cl in selected_cell_lines for dt in dt_data[cl]]
            all_passages = [list(range(passages)) for passages in max_passages]
            flat_passages = [item for sublist in all_passages for item in sublist]
            all_cell_lines = [[cl] * len(pd_data[cl]) for cl in selected_cell_lines]
            flat_cell_lines = [item for sublist in all_cell_lines for item in sublist]

            df_calculated = pd.DataFrame({
                'Cell Line': flat_cell_lines,
                'Passage Number': flat_passages,
                'Population Doublings': all_pd,
                'Time in Culture (hrs)': all_times,
                'Doubling Time (hrs/PD)': all_dts
            })
            st.dataframe(df_calculated)  # Display the DataFrame in the Streamlit interface
            plot_graphs(cpd_data, dt_data)

    if st.session_state['graph'].nodes():
        draw_graph()

if __name__ == "__main__":
    main()
