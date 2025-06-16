import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import ast
from collections import defaultdict
import numpy as np
import requests
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="PDF Screening Criteria Dashboard",
    page_icon="📄",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stCheckbox {
        margin: 0.5rem 0;
    }
    .criteria-group {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    /* Target all expanders and then filter by content */
    .streamlit-expanderHeader {
        background-color: #fff5f5 !important;
    }
    
    /* Or target the expander container */
    div.streamlit-expander {
        background-color: #fff5f5 !important;
        border: 1px solid #ffe0e0 !important;
    }
    
    /* Target using data-testid with descendant selector */
    div[data-testid="column"] div[data-testid="stExpander"] {
        background-color: #fff5f5 !important;
    }
    
    /* Target the details element inside expander */
    details {
        background-color: #fff5f5 !important;
    }
</style>
""", unsafe_allow_html=True)


# Sankey chart functions
@st.cache_data
def load_sankey_data():
    """Load and process the Excel file for Sankey diagram"""
    try:
        # Read the Excel file for sankey chart
        df = pd.read_excel('screening_classification_criteria.xlsx')
        
        # Clean the data - remove rows where Category is NaN
        df = df.dropna(subset=['Category'])
        
        return df
    except FileNotFoundError:
        st.warning("Sankey data file 'screening_classification_criteria.xlsx' not found. Sankey chart will be skipped.")
        return None
    except Exception as e:
        st.warning(f"Error loading sankey data: {str(e)}")
        return None

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def create_sankey_data(df):
    """Create nodes and links for the Sankey diagram"""
    
    # Darker color mapping for action types
    colors = {
        'Needs Summary': '#a8d5a8',     # Darker green
        'Record for Later': '#f0d966',  # Darker yellow
        'No Action': '#e89ea3'          # Darker red
    }
    
    nodes = []
    links = []
    node_colors = []
    # Add positioning arrays for better control
    node_x = []
    node_y = []
    
    # Level 0: Root node
    nodes.append('Screening Criterias')
    node_colors.append('#FFB7B2')  # Soft pink for root
    node_x.append(0.01)  # Far left
    node_y.append(0.5)   # Center vertically
    
    # Level 1: Categories
    categories = df['Category'].unique()
    cat_count = len(categories)
    # Calculate compact spacing for categories
    cat_spacing = 0.4 / cat_count  # Use only 60% of vertical space
    cat_start_y = 0.3  # Start at 20% from top
    for i, cat in enumerate(categories):
        nodes.append(cat)
        node_colors.append('#FFB7B2')  # Soft pink for categories
        node_x.append(0.20)  # Position categories
        node_y.append(cat_start_y + i * cat_spacing)  # Compact vertical distribution
        
        # Add link from root to category
        links.append({
            'source': 0,  # Root node index
            'target': len(nodes) - 1,
            'value': 1,
            'color': '#f8d6d5'  # Light pink for root to category links
        })
    
    # Level 2: Specific Criteria
    criteria_start_idx = len(nodes)
    criteria_count = len(df)
    # Calculate compact spacing for criteria
    criteria_spacing = 0.6 / criteria_count  # Use only 70% of vertical space
    criteria_start_y = 0.20  # Start at 15% from top
    for i, (_, row) in enumerate(df.iterrows()):
        criteria_name = row['Specific Criteria']  
        nodes.append(criteria_name)
        node_colors.append('#808080')  # Gray for criteria
        node_x.append(0.35)  # Position criteria in the middle
        node_y.append(criteria_start_y + i * criteria_spacing)  # Compact vertical distribution
        
        # Add link from Category to Specific Criteria
        cat_idx = list(categories).index(row['Category']) + 1  # +1 because of root node
        criteria_idx = len(nodes) - 1
        links.append({
            'source': cat_idx,
            'target': criteria_idx,
            'value': 1,
            'color': '#f8d6d5'  # Light pink for category to criteria links
        })
    
    # Level 3: Action Values
    action_start_idx = len(nodes)
    action_nodes_added = []
    action_node_order = []  # Track order of action nodes
    
    # Process each row to create action value nodes and links
    for row_idx, row in df.iterrows():
        criteria_idx = criteria_start_idx + row_idx
        
        # Process each action column in order
        for action_col in ['Needs Summary', 'Record for Later', 'No Action']:
            value = row[action_col]
            
            # Skip if value is NaN, False, or None
            if pd.isna(value) or value is False or value is None:
                continue
            
            # Handle different value types
            if value is True:
                # For boolean True values, use a descriptive name
                node_name = f"{row['Specific Criteria']} (Yes)"
            elif isinstance(value, str):
                # Use the entire string as a single node (don't split)
                node_name = value.strip().strip('"')
            else:
                continue
            
            # Skip empty values
            if not node_name:
                continue
                
            # Check if this node already exists
            if node_name not in nodes:
                nodes.append(node_name)
                node_colors.append(colors[action_col])
                action_nodes_added.append(node_name)
                action_node_order.append((node_name, action_col))
                # Position action nodes NOT at the far right edge
                node_x.append(0.60)  # Move left from 0.85 to leave space for labels
                # We'll set Y positions after all nodes are added
                node_y.append(0.5)  # Temporary position
            
            action_node_idx = nodes.index(node_name)
            
            # Add link from criteria to action value
            # Convert hex to rgba for transparency
            r, g, b = hex_to_rgb(colors[action_col])
            rgba_color = f'rgba({r},{g},{b},0.5)'
            
            links.append({
                'source': criteria_idx,
                'target': action_node_idx,
                'value': 1,
                'color': rgba_color
            })
    
    # Create dummy nodes to force labels on the right
    dummy_nodes_start = len(nodes)
    for i, (node_name, action_col) in enumerate(action_node_order):
        # Add invisible dummy node
        nodes.append('')  # Empty label
        node_colors.append('rgba(0,0,0,0)')  # Transparent
        node_x.append(0.99)  # Far right
        node_y.append((i + 0.5) / len(action_node_order))
        
        # Add invisible link from action node to dummy
        action_idx = nodes.index(node_name)
        links.append({
            'source': action_idx,
            'target': len(nodes) - 1,
            'value': 0.001,  # Very small value to make it nearly invisible
            'color': 'rgba(0,0,0,0)'  # Transparent
        })
    
    # Update Y positions for action nodes to match their order
    action_node_count = len(action_nodes_added)
    for i, (node_name, _) in enumerate(action_node_order):
        node_idx = nodes.index(node_name)
        node_y[node_idx] = (i + 0.5) / action_node_count
    
    return nodes, links, node_colors, node_x, node_y

def create_sankey_chart(sankey_df):
    """Create the Sankey chart"""
    if sankey_df is None:
        return None
    
    try:
        # Create Sankey data
        nodes, links, node_colors, node_x, node_y = create_sankey_data(sankey_df)
        
        # Create Sankey diagram with explicit positioning
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=20,  # Padding between nodes
                thickness=12,  # Node thickness
                line=dict(color="rgba(0,0,0,0)", width=0),  # Transparent border
                label=nodes,
                color=node_colors,
                x=node_x,  # Explicit X positioning
                y=node_y,  # Explicit Y positioning
                hovertemplate="<extra></extra>"  # Disable hover for nodes
            ),
            link=dict(
                source=[link['source'] for link in links],
                target=[link['target'] for link in links],
                value=[link['value'] for link in links],
                color=[link['color'] for link in links],
                hovertemplate="<extra></extra>"  # Disable hover for links
            ),
            textfont=dict(
                color="black",
                size=12,  # Slightly smaller for dashboard integration
                family="Arial, sans-serif"
            ),
            hoverinfo="none"  # Disable all hover info
        )])
        
        fig.update_layout(
            title=dict(
                text="Screening Classification Flow",
                font=dict(size=18, weight='bold'),
                x=0.5,
                xanchor='center'
            ),
            font=dict(
                size=12,
                color="black", 
                family="Arial, sans-serif"
            ),
            height=600,  # Adjusted for dashboard integration
            margin=dict(l=50, r=50, t=60, b=30),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False
        )
        
        # Update traces to ensure no text shadows
        fig.update_traces(
            textfont=dict(
                color="black",
                size=12,
                family="Arial, sans-serif"
            ),
            selector=dict(type='sankey')
        )
        
        return fig
    except Exception as e:
        st.warning(f"Error creating Sankey chart: {str(e)}")
        return None
    

def convert_back_to_original_format(transposed_df):
    """Convert the transposed dataframe back to original Excel format with proper data types"""
    
    # Read the original file to get the exact original order AND original data types
    original_screening_data = pd.read_excel('screening_criteria_dataset.xlsx')
    
    # Create a copy of the transposed dataframe
    df_copy = transposed_df.copy()
    
    # Set PDF column as index
    df_copy = df_copy.set_index('PDF')
    
    # Transpose back
    original_df = df_copy.T
    
    # Split the column headers back to Category and Specific Criteria
    categories = []
    criteria = []
    
    for col_header in original_df.index:
        if ' - ' in col_header:
            cat, crit = col_header.split(' - ', 1)
            categories.append(cat)
            criteria.append(crit)
        else:
            categories.append('')
            criteria.append(col_header)
    
    # Reset index and add Category and Specific Criteria columns
    original_df = original_df.reset_index(drop=True)
    original_df.insert(0, 'Category', categories)
    original_df.insert(1, 'Specific Criteria', criteria)
    
    # Create a mapping of (Category, Specific Criteria) to original row data
    original_lookup = {}
    for idx, row in original_screening_data.iterrows():
        key = (row['Category'], row['Specific Criteria'])
        original_lookup[key] = row
    
    # Create a list to store the reordered dataframe with restored data types
    reordered_rows = []
    
    # Go through the original order and restore data types
    for idx, orig_row in original_screening_data.iterrows():
        orig_key = (orig_row['Category'], orig_row['Specific Criteria'])
        
        # Find this row in our converted dataframe
        found_row = None
        for conv_idx, conv_row in original_df.iterrows():
            conv_key = (conv_row['Category'], conv_row['Specific Criteria'])
            
            if orig_key == conv_key:
                found_row = conv_row.copy()
                break
        
        if found_row is not None:
            # Restore original data types for all PDF columns
            pdf_columns = [col for col in found_row.index if col not in ['Category', 'Specific Criteria']]
            
            for pdf_col in pdf_columns:
                if pdf_col in orig_row.index:
                    original_value = orig_row[pdf_col]
                    current_value = found_row[pdf_col]
                    
                    # Only restore if the original value was not a simple string/number/boolean
                    # and if the current value has been modified from its original form
                    if should_restore_original_value(original_value, current_value, orig_row['Specific Criteria']):
                        found_row[pdf_col] = original_value
                    else:
                        # Keep the potentially edited value (like rejection reasons)
                        found_row[pdf_col] = current_value
            
            reordered_rows.append(found_row)
    
    # Convert back to dataframe with original order and data types
    if reordered_rows:
        final_df = pd.DataFrame(reordered_rows)
        final_df = final_df.reset_index(drop=True)
        return final_df
    else:
        # Fallback to the original approach if matching fails
        return original_df


def should_restore_original_value(original_value, current_value, criteria_name):
    """
    Determine if we should restore the original value or keep the current (potentially edited) value
    """
    # Always allow editing of rejection reasons
    if 'rejection' in criteria_name.lower():
        return False
    
    # If the original value was a list or dict, restore it unless it was intentionally edited
    if isinstance(original_value, (list, dict)):
        return True
    
    # If the original was a number and current is a string representation, restore the number
    if isinstance(original_value, (int, float)) and isinstance(current_value, str):
        try:
            # Check if the string is just a string representation of the number
            if str(original_value) == current_value:
                return True
        except:
            pass
    
    # For boolean values that got converted to strings
    if isinstance(original_value, bool) and isinstance(current_value, str):
        if str(original_value) == current_value:
            return True
    
    # If values are essentially the same, keep original type
    if str(original_value) == str(current_value):
        return True
    
    # Otherwise, keep the current value (it might have been intentionally edited)
    return False


def format_competitor_products_for_display(value):
    """Format competitor products dictionary into readable format for dashboard display only"""
    if pd.isna(value) or str(value) in ['{}', '', 'nan']:
        return ''
    
    try:
        # Company name abbreviations
        company_abbrev = {
            'Philip Morris International': 'PMI',
            'British American Tobacco': 'BAT', 
            'Japan Tobacco International': 'JTI',
            'Imperial Brands': 'Imperial',
            'Reynolds American': 'RAI'
        }
        
        # Parse the value if it's a string representation of a dict
        if isinstance(value, str):
            # Handle cases where it might be a dict string
            if value.startswith('{') and value.endswith('}'):
                try:
                    parsed_value = ast.literal_eval(value)
                except:
                    try:
                        parsed_value = json.loads(value.replace("'", '"'))
                    except:
                        return str(value)
            else:
                return str(value)
        elif isinstance(value, dict):
            parsed_value = value
        else:
            return str(value)
        
        # Format the dictionary
        formatted_parts = []
        for company, products in parsed_value.items():
            # Get abbreviation or use original name
            abbrev = company_abbrev.get(company, company)
            
            # Handle products list
            if isinstance(products, list):
                products_str = ', '.join([str(p) for p in products if p])
            else:
                products_str = str(products)
            
            if products_str:
                formatted_parts.append(f"{abbrev}: {products_str}")
        
        return '; '.join(formatted_parts)
        
    except Exception as e:
        # If parsing fails, return the original value as string
        return str(value) if str(value) != '{}' else ''


def create_display_dataframe(screening_data):
    """Create a properly formatted dataframe for display that preserves original data for saving"""
    # Transpose the data so criteria are columns and PDFs are rows
    transposed_data = screening_data.set_index(['Category', 'Specific Criteria']).T
    
    # Create multi-level column headers
    transposed_data.columns = [f"{cat} - {crit}" for cat, crit in transposed_data.columns]
    
    # Reset index to make PDF names a column
    transposed_data = transposed_data.reset_index()
    transposed_data = transposed_data.rename(columns={'index': 'PDF'})
    
    # Create a display version with formatting
    display_data = transposed_data.copy()
    
    # Clean up list formatting for display - but keep original data separate
    for col in display_data.columns:
        if col != 'PDF':  # Don't modify the PDF column
            if 'Competitor Products' in col:
                # Special handling for competitor products dictionary format
                display_data[col] = display_data[col].apply(lambda x: format_competitor_products_for_display(x))
            else:
                # General formatting for other columns - only for display
                display_data[col] = display_data[col].apply(lambda x: 
                    str(x).replace('[', '').replace(']', '').replace("'", '') if pd.notna(x) and str(x) not in ['{}', ''] else 
                    '' if str(x) == '{}' else x)
    
    # Reorder columns to move Classification System columns next to PDF
    all_columns = list(display_data.columns)
    pdf_col = ['PDF']
    classification_cols = [col for col in all_columns if 'Classification System' in col]
    other_cols = [col for col in all_columns if col != 'PDF' and 'Classification System' not in col]
    
    # New column order: PDF, Classification System columns, then all others
    new_column_order = pdf_col + classification_cols + other_cols
    display_data = display_data[new_column_order]
    
    # CRITICAL FIX: Don't change the index for display_data to avoid misalignment
    # Keep the same index as the original transposed_data (0, 1, 2, ...)
    # We'll handle the numbering in the Streamlit display instead
    
    return display_data, transposed_data  # Return both display and original data


def save_to_github_direct(df):
    """Save dataframe directly to GitHub file (requires write access)"""
    
    # GitHub configuration - add these to your secrets or environment
    GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")  # Your GitHub personal access token
    REPO_OWNER = "apratim-sigmoid"  # Replace with your GitHub username
    REPO_NAME = "IB-GenAI-RnD-Screening-Criteria"  # Replace with your repository name
    FILE_PATH = "screening_criteria_dataset.xlsx"  # Path to your Excel file
    BRANCH = "main"  # or "master" depending on your default branch
    
    if not GITHUB_TOKEN:
        st.error("GitHub token not found. Please add GITHUB_TOKEN to your secrets.")
        return False
    
    try:
        # Convert dataframe back to Excel format
        output = BytesIO()
        
        # You'll need to convert the transposed data back to original format
        # This is a simplified version - you may need to adjust based on your exact data structure
        original_format_df = convert_back_to_original_format(df)
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            original_format_df.to_excel(writer, index=False)
        
        # Encode file content as base64
        file_content = base64.b64encode(output.getvalue()).decode()
        
        # Get current file SHA (required for updates)
        url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            current_sha = response.json()["sha"]
        else:
            st.error(f"Could not get current file SHA: {response.status_code}")
            return False
        
        # Update file
        update_data = {
            "message": f"Update screening data - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "content": file_content,
            "sha": current_sha,
            "branch": BRANCH
        }
        
        response = requests.put(url, json=update_data, headers=headers)
        
        if response.status_code == 200:
            return True
        else:
            st.error(f"Failed to update file: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        st.error(f"Error in GitHub save: {str(e)}")
        return False
        

def load_data():
    """Load and process the screening criteria dataset"""
    # Read the Excel file as screening_data (includes metadata)
    screening_data = pd.read_excel('screening_criteria_dataset.xlsx')
    
    # Create df by excluding metadata AND classification system categories from analysis
    df = screening_data[
        (screening_data['Category'] != 'Metadata') & 
        (screening_data['Category'] != 'Classification System')
    ].copy()
    
    # Get PDF names (all columns except first two)
    pdf_columns = df.columns[2:].tolist()
    
    # Process the data into a structured format
    criteria_data = []
    for idx, row in df.iterrows():
        criteria_data.append({
            'category': row['Category'],
            'criteria': row['Specific Criteria'],
            'values': row[pdf_columns].tolist(),
            'pdf_columns': pdf_columns
        })
    
    
    return df, criteria_data, pdf_columns, screening_data

def parse_value(value):
    """Parse different types of values in the dataset"""
    if pd.isna(value):
        return None
    
    # Handle string boolean values
    if isinstance(value, str):
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        # Try to parse as list
        elif value.startswith('['):
            try:
                return ast.literal_eval(value)
            except:
                return value
        # Try to parse as dict
        elif value.startswith('{'):
            try:
                return json.loads(value.replace("'", '"'))
            except:
                try:
                    return ast.literal_eval(value)
                except:
                    return value
    
    # Return as is for other types
    return value

def extract_sample_sizes(criteria_data):
    """Extract all sample sizes from the dataset to determine slider range"""
    sample_sizes = []
    
    # Find sample size criteria
    for criteria in criteria_data:
        if 'sample size' in criteria['criteria'].lower():
            for value in criteria['values']:
                parsed_value = parse_value(value)
                if isinstance(parsed_value, (int, float)) and not pd.isna(parsed_value):
                    sample_sizes.append(int(parsed_value))
                elif isinstance(parsed_value, str):
                    # Try to extract numbers from string
                    import re
                    numbers = re.findall(r'\d+', str(parsed_value))
                    for num in numbers:
                        try:
                            sample_sizes.append(int(num))
                        except:
                            continue
    
    return sample_sizes

def get_pdfs_matching_criteria(criteria_data, criteria_index, sample_size_range=None):
    """Get list of PDFs matching a specific criteria"""
    matching_pdfs = []
    criteria = criteria_data[criteria_index]
    
    # List of criteria that have boolean values
    boolean_criteria = [
        'Competitor-funded research',
        'Priority health endpoints',
        'Nicotine general studies',
        'Peer review status',
        'Novel/interesting findings'
    ]
    
    # Check if this is a sample size criteria
    is_sample_size_criteria = 'sample size' in criteria['criteria'].lower()
    
    for i, value in enumerate(criteria['values']):
        parsed_value = parse_value(value)
        pdf_name = criteria['pdf_columns'][i]
        
        # Handle sample size criteria with range
        if is_sample_size_criteria and sample_size_range is not None:
            sample_size = None
            if isinstance(parsed_value, (int, float)) and not pd.isna(parsed_value):
                sample_size = int(parsed_value)
            elif isinstance(parsed_value, str):
                import re
                numbers = re.findall(r'\d+', str(parsed_value))
                if numbers:
                    try:
                        sample_size = int(numbers[0])  # Take first number found
                    except:
                        continue
            
            if sample_size is not None:
                # If max range is 10000, include all values >= 10000
                if sample_size_range[1] == 10000 and sample_size >= sample_size_range[0]:
                    matching_pdfs.append(pdf_name)
                # Otherwise use strict range
                elif sample_size_range[0] <= sample_size <= sample_size_range[1]:
                    matching_pdfs.append(pdf_name)
        
        # For non-sample size criteria, use original logic
        elif not is_sample_size_criteria:
            # For boolean criteria, only consider True values
            if criteria['criteria'] in boolean_criteria:
                if isinstance(parsed_value, bool) and parsed_value:
                    matching_pdfs.append(pdf_name)
                elif isinstance(parsed_value, str) and parsed_value.lower() == 'true':
                    matching_pdfs.append(pdf_name)
            # For other criteria, check different data types
            elif isinstance(parsed_value, bool) and parsed_value:
                matching_pdfs.append(pdf_name)
            elif isinstance(parsed_value, list) and len(parsed_value) > 0:
                matching_pdfs.append(pdf_name)
            elif isinstance(parsed_value, dict) and len(parsed_value) > 0:
                matching_pdfs.append(pdf_name)
            elif isinstance(parsed_value, str) and parsed_value not in ['', 'Unknown', 'No Action']:
                matching_pdfs.append(pdf_name)
    
    return matching_pdfs

def create_progress_bar(selected_count, total_count):
    """Create a horizontal progress bar"""
    percentage = (selected_count / total_count * 100) if total_count > 0 else 0
    
    # Create a horizontal bar chart
    fig = go.Figure()
    
    # Add the selected papers bar
    fig.add_trace(go.Bar(
        x=[selected_count],
        y=['Papers'],
        name='Selected',
        orientation='h',
        marker=dict(color='#f07300'),  # Updated to use the specified orange color
        text=f'{selected_count} ({percentage:.1f}%)',
        textposition='inside',
        textfont=dict(color='white', size=14, weight='bold'),
        hovertemplate='Selected: %{x} papers<extra></extra>'
    ))
    
    # Add the remaining papers bar
    remaining = total_count - selected_count
    fig.add_trace(go.Bar(
        x=[remaining],
        y=['Papers'],
        name='Remaining',
        orientation='h',
        marker=dict(color='lightgray'),
        text=f'{remaining} remaining',
        textposition='inside',
        textfont=dict(color='black', size=12),
        hovertemplate='Remaining: %{x} papers<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'<b>Selected Papers Progress: {selected_count} / {total_count}</b>',
            font=dict(size=18),
            x=0.5,
            xanchor='center'
        ),
        barmode='stack',
        height=150,
        showlegend=False,
        xaxis=dict(
            range=[0, total_count],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            title='Number of Papers',
            title_font=dict(size=12)
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False
        ),
        margin=dict(l=20, r=20, t=60, b=40),
        plot_bgcolor='white'
    )
    
    # Add annotations for better clarity
    fig.add_annotation(
        x=selected_count/2,
        y=0,
        text=f'<b>{selected_count}</b>',
        showarrow=False,
        font=dict(size=20, color='white'),
        yshift=0
    )
    
    fig.add_annotation(
        x=selected_count + remaining/2,
        y=0,
        text=f'<b>{remaining}</b>',
        showarrow=False,
        font=dict(size=16, color='black'),
        yshift=0
    )
    
    return fig


def create_category_distribution(criteria_data, selected_indices, pdf_columns, sample_size_range=None):
    """Create a horizontal bar chart showing PDF distribution by criteria"""
    # Prepare data for visualization
    labels = []
    total_values = []
    selected_values = []
    colors = []
    
    # Color palette for categories
    category_colors = {
        'Product Mentions': '#1f77b4',      # Blue
        'Funding Sources': '#ff7f0e',       # Orange
        'Health Endpoints': '#2ca02c',      # Green
        'Product Categories': '#d62728',    # Red
        'Study Quality': '#9467bd',         # Purple
        'Others': '#e377c2',                # Pink
    }
    
    # Add all criteria in original order (as they appear in the data)
    for idx, criteria in enumerate(criteria_data):
        category = criteria['category']
        criteria_name = criteria['criteria']
        
        # Handle sample size criteria differently
        if 'sample size' in criteria_name.lower():
            matching_pdfs = get_pdfs_matching_criteria(criteria_data, idx, sample_size_range)
        else:
            matching_pdfs = get_pdfs_matching_criteria(criteria_data, idx)
        
        labels.append(criteria_name)
        total_values.append(len(matching_pdfs))
        selected_values.append(len(matching_pdfs) if idx in selected_indices else 0)
        
        # Get color for this category
        color = category_colors.get(category, '#7f7f7f')  # Default gray if category not found
        colors.append(color)
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    # Add total PDFs bars
    fig.add_trace(go.Bar(
        name='Total Available',
        y=labels,
        x=total_values,
        orientation='h',
        marker=dict(
            color=colors,
            opacity=0.3,
            line=dict(color='rgba(0,0,0,0.3)', width=1)
        ),
        text=[f'{v}' for v in total_values],
        textposition='outside',
        textfont=dict(size=10),
        hovertemplate='%{y}<br>Total: %{x} PDFs<extra></extra>'
    ))
    
    # Add selected PDFs bars - use same colors
    fig.add_trace(go.Bar(
        name='Selected',
        y=labels,
        x=selected_values,
        orientation='h',
        marker=dict(
            color=colors,
            opacity=1.0,
            line=dict(color='rgba(0,0,0,0.5)', width=1)
        ),
        text=[f'{v}' if v > 0 else '' for v in selected_values],
        textposition='inside',
        textfont=dict(color='white', size=10, weight='bold'),
        hovertemplate='%{y}<br>Selected: %{x} PDFs<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="PDF Distribution by Screening Criteria",
            font=dict(size=18, weight='bold'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="Number of PDFs",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            range=[0, max(total_values) * 1.1] if total_values else [0, 1]  # Add some padding
        ),
        yaxis=dict(
            title="",
            autorange='reversed',  # Show from top to bottom
            tickfont=dict(size=11)
        ),
        barmode='overlay',
        height=max(600, len(labels) * 30),  # Dynamic height based on number of items
        margin=dict(l=300, r=100, t=80, b=50),  # Larger left margin for labels
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        plot_bgcolor='white'
    )
    
    return fig

def create_incremental_impact_chart(criteria_data, selected_indices, pdf_columns, sample_size_range=None):
    """Create a horizontal bar chart showing incremental impact of unselected criteria in same order as main chart"""
    
    # Get currently selected PDFs
    current_pdfs = set()
    for idx in selected_indices:
        if 'sample size' in criteria_data[idx]['criteria'].lower():
            current_pdfs.update(get_pdfs_matching_criteria(criteria_data, idx, sample_size_range))
        else:
            current_pdfs.update(get_pdfs_matching_criteria(criteria_data, idx))
    
    # Color palette for categories (same as main chart)
    category_colors = {
        'Product Mentions': '#1f77b4',      # Blue
        'Funding Sources': '#ff7f0e',       # Orange
        'Health Endpoints': '#2ca02c',      # Green
        'Product Categories': '#d62728',    # Red
        'Study Quality': '#9467bd',         # Purple
        'Others': '#e377c2',                # Pink
    }
    
    # Prepare data for unselected criteria only, in original order
    labels = []
    incremental_values = []
    colors = []
    
    for idx, criteria in enumerate(criteria_data):
        # Only include criteria that are NOT selected
        if idx not in selected_indices:
            category = criteria['category']
            criteria_name = criteria['criteria']
            
            # Get matching PDFs for this criteria
            if 'sample size' in criteria_name.lower():
                matching_pdfs = get_pdfs_matching_criteria(criteria_data, idx, sample_size_range)
            else:
                matching_pdfs = get_pdfs_matching_criteria(criteria_data, idx)
            
            # Calculate incremental impact (new PDFs this criteria would add)
            new_pdfs = set(matching_pdfs) - current_pdfs
            incremental = len(new_pdfs)
            
            # Only include if there's some incremental value
            if incremental > 0:
                labels.append(criteria_name)
                incremental_values.append(incremental)
                
                # Get color for this category
                color = category_colors.get(category, '#7f7f7f')  # Default gray if category not found
                colors.append(color)
    
    if not labels:
        return None
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    # Add incremental impact bars
    fig.add_trace(go.Bar(
        name='New PDFs',
        y=labels,
        x=incremental_values,
        orientation='h',
        marker=dict(
            color=colors,
            opacity=1.0,
            line=dict(color='rgba(0,0,0,0.5)', width=1)
        ),
        text=[f'+{v}' for v in incremental_values],
        textposition='inside',
        textfont=dict(color='white', size=10, weight='bold'),
        hovertemplate='%{y}<br>New PDFs: +%{x}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="Incremental Impact of Unselected Criteria in PDF Selection",
            font=dict(size=18, weight='bold'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="Number of New PDFs",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            range=[0, max(incremental_values) * 1.1] if incremental_values else [0, 1]
        ),
        yaxis=dict(
            title="",
            autorange='reversed',  # Show from top to bottom (same as main chart)
            tickfont=dict(size=11)
        ),
        height=max(400, len(labels) * 35),  # Dynamic height: minimum 400px, 35px per bar
        margin=dict(l=300, r=100, t=80, b=50),  # Larger left margin for labels
        showlegend=False,
        plot_bgcolor='white'
    )
    
    return fig


def create_product_distribution(criteria_data, selected_indices=None):
    """Create visualization for product mentions - shows all data regardless of selection"""
    imperial_products = defaultdict(int)
    competitor_products = defaultdict(lambda: defaultdict(int))
    
    # Find product mention criteria and process ALL data (not just selected)
    for idx, criteria in enumerate(criteria_data):
        if criteria['criteria'] == 'Imperial Brands Products':
            for i, value in enumerate(criteria['values']):
                # Skip completely empty values
                if pd.isna(value):
                    continue
                
                parsed = parse_value(value)
                
                # Handle list of products (this is the main case we expect)
                if isinstance(parsed, list) and len(parsed) > 0:
                    for product in parsed:
                        if product and str(product).strip():
                            clean_product = str(product).strip()
                            # Remove any brackets or quotes that might be present
                            clean_product = clean_product.replace('[', '').replace(']', '').replace('"', '').replace("'", "")
                            if clean_product:  # Only add non-empty products
                                imperial_products[clean_product] += 1
                
                # Handle single product as string (fallback case)
                elif isinstance(parsed, str) and parsed.strip() and parsed.strip() != '[]':
                    clean_product = parsed.strip()
                    if clean_product not in ['Unknown', 'No Action', '[]', '']:
                        # If it looks like a list string, try to parse it further
                        if clean_product.startswith('[') and clean_product.endswith(']'):
                            # Remove brackets and split by comma
                            inner_content = clean_product[1:-1]
                            if inner_content.strip():
                                products = [p.strip().replace('"', '').replace("'", "") for p in inner_content.split(',')]
                                for product in products:
                                    if product and product.strip():
                                        imperial_products[product.strip()] += 1
                        else:
                            imperial_products[clean_product] += 1
        
        elif criteria['criteria'] == 'Competitor Products':
            for i, value in enumerate(criteria['values']):
                parsed = parse_value(value)
                # Only process if we have actual data
                if pd.isna(value) or value in ['', 'Unknown', 'No Action', 'False', 'false', None]:
                    continue
                    
                if isinstance(parsed, dict):
                    for company, products in parsed.items():
                        if isinstance(products, list):
                            for product in products:
                                if product and str(product).strip() and str(product).strip() != '':
                                    clean_product = str(product).strip().replace('"', '').replace("'", "")
                                    competitor_products[company][clean_product] += 1
    
    # Create combined visualization
    figs = []
    
    # Combine all products data
    if imperial_products or competitor_products:
        # Prepare data for combined chart
        all_companies = []
        all_products = []
        all_counts = []
        
        # Add Imperial Brands data
        if imperial_products:
            for product, count in imperial_products.items():
                all_companies.append('Imperial Brands')
                all_products.append(product)
                all_counts.append(count)
        
        # Add competitor data
        if competitor_products:
            for company, products in competitor_products.items():
                for product, count in products.items():
                    all_companies.append(company)
                    all_products.append(product)
                    all_counts.append(count)
        
        if all_companies:
            # Create DataFrame for the combined chart
            df_combined = pd.DataFrame({
                'Company': all_companies,
                'Product': all_products,
                'Count': all_counts
            })
            
            # Keep Imperial Brands first, then other companies (no sorting by count)
            # The order is already correct from how we built the data
            
            # Create custom color mapping
            unique_companies = df_combined['Company'].unique()
            color_map = {}
            
            # Assign specific colors to known companies
            company_colors = {
                'Imperial Brands': '#f07300',  # Orange
                'Philip Morris International': "#a81717",  # Red
                'Japan Tobacco International': '#2ca02c',             # Green
                'British American Tobacco': '#1f77b4'  # Blue
            }
            
            # Assign colors to companies
            other_colors = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            color_idx = 0
            
            for company in unique_companies:
                if company in company_colors:
                    color_map[company] = company_colors[company]
                else:
                    # For any other companies not specified
                    color_map[company] = other_colors[color_idx % len(other_colors)]
                    color_idx += 1
            
            # Create the combined bar chart
            fig = px.bar(
                df_combined, 
                x='Product', 
                y='Count', 
                color='Company',
                title="Imperial Brands and Competitor Products Distribution",
                labels={'Count': 'Number of Papers', 'Product': 'Products'},
                color_discrete_map=color_map
            )
            
            # Update layout for better readability
            fig.update_layout(
                xaxis_tickangle=-45,
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                # Keep original order - Imperial Brands first, then competitors
                xaxis=dict(categoryorder='array', categoryarray=df_combined['Product'].tolist())
            )
            
            figs.append(fig)
    
    # Create separate pie chart for Imperial Brands if there are products
    if imperial_products:
        # Create different shades of the Imperial orange color for pie chart
        num_products = len(imperial_products)
        base_color = '#f07300'
        
        # Generate different shades of orange
        import colorsys
        
        # Convert hex to RGB
        base_rgb = tuple(int(base_color[i:i+2], 16) for i in (1, 3, 5))
        base_hsl = colorsys.rgb_to_hls(base_rgb[0]/255, base_rgb[1]/255, base_rgb[2]/255)
        
        # Generate only lighter shades by blending with white
        colors = []
        for i in range(num_products):
            if i == 0:
                # First product uses the exact base color
                colors.append(base_color)
            else:
                # Create lighter versions by blending with white
                # Convert base color to RGB
                base_r, base_g, base_b = base_rgb
                
                # Blend factor: 0 = original color, 1 = white
                blend_factor = 0.4 * i / max(1, num_products - 1)  # Up to 40% white blending
                
                # Blend with white (255, 255, 255)
                new_r = int(base_r + (255 - base_r) * blend_factor)
                new_g = int(base_g + (255 - base_g) * blend_factor)
                new_b = int(base_b + (255 - base_b) * blend_factor)
                
                hex_color = '#{:02x}{:02x}{:02x}'.format(new_r, new_g, new_b)
                colors.append(hex_color)
        
        fig_pie = px.pie(
            values=list(imperial_products.values()),
            names=list(imperial_products.keys()),
            title="Imperial Brands Product Distribution",
            color_discrete_sequence=colors
        )
        
        # Update pie chart layout
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=500)
        
        figs.append(fig_pie)
    
    return figs


# Main app
def main():
    st.title("📄 PDF Screening Criteria Dashboard")
    
    # Load data
    df, criteria_data, pdf_columns, screening_data = load_data()
    total_pdfs = len(pdf_columns)

    # Load Sankey data
    sankey_df = load_sankey_data()
    
    # Extract sample sizes to determine slider range
    sample_sizes = extract_sample_sizes(criteria_data)
    min_sample_size = min(sample_sizes) if sample_sizes else 0
    max_sample_size = max(sample_sizes) if sample_sizes else 10000
    
    # Initialize session state for checkboxes if not exists
    if 'selected_criteria' not in st.session_state:
        st.session_state.selected_criteria = set()
    
    # Initialize sample size range in session state
    if 'sample_size_range' not in st.session_state:
        st.session_state.sample_size_range = (min_sample_size, 10000)
    
    # Create sidebar for criteria selection
    with st.sidebar:
        st.header("Screening Criteria")
        
        # Quick actions at the top
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Select All", key="select_all"):
                # Select all criteria indices
                st.session_state.selected_criteria = set(range(len(criteria_data)))
                st.rerun()
        with col2:
            if st.button("Clear All", key="clear_all"):
                # Clear all selections
                st.session_state.selected_criteria = set()
                st.rerun()
                
        # Group criteria by category
        categories = {}
        for idx, criteria in enumerate(criteria_data):
            category = criteria['category']
            if category not in categories:
                categories[category] = []
            categories[category].append((idx, criteria['criteria']))
        
        # Display checkboxes grouped by category
        selected_indices = []
        sample_size_idx = None
        sample_size_range = None
        
        # Find sample size criteria index
        for idx, criteria in enumerate(criteria_data):
            if 'sample size' in criteria['criteria'].lower():
                sample_size_idx = idx
                break
        
        # PRE-CALCULATE current selected PDFs for incremental impact calculation
        current_selected = set()
        for idx in st.session_state.selected_criteria:
            if 'sample size' in criteria_data[idx]['criteria'].lower():
                current_selected.update(get_pdfs_matching_criteria(criteria_data, idx, st.session_state.sample_size_range))
            else:
                current_selected.update(get_pdfs_matching_criteria(criteria_data, idx))
        
        for category, criteria_list in categories.items():
            st.subheader(category)
            for idx, criteria_name in criteria_list:
                # Handle sample size criteria with slider
                if 'sample size' in criteria_name.lower():
                    # Create double range slider first
                    sample_size_range = st.slider(
                        f"**📊 {criteria_name}**",
                        min_value=min_sample_size,
                        max_value=10000,
                        value=st.session_state.sample_size_range,
                        step=10,
                        help=f"Filter studies by sample size range. Values ≥10000 are included when max is set to 10000",
                        format="%d",
                        key=f'sample_size_range_{idx}'
                    )
                    
                    # Update session state
                    st.session_state.sample_size_range = sample_size_range
                    
                    # Calculate matching PDFs for the current range and display count
                    matching_sample_pdfs = get_pdfs_matching_criteria(criteria_data, idx, sample_size_range)
                    st.markdown(f"**({len(matching_sample_pdfs)} PDFs)**")
                    
                    # Show range display with 10000+ for max value
                    max_display = "10000+" if sample_size_range[1] == 10000 else str(sample_size_range[1])
                    st.markdown(f"Range: {sample_size_range[0]} - {max_display}")
                    
                    # Add sample size criteria to selection if range is modified from default or if it's in session state
                    if (sample_size_range != (min_sample_size, 10000) or 
                        sample_size_range[0] > min_sample_size or 
                        sample_size_range[1] < 10000 or 
                        idx in st.session_state.selected_criteria):
                        selected_indices.append(idx)
                    
                    continue  # Skip the checkbox for sample size
                
                # Calculate how many PDFs match this criteria
                matching_pdfs = get_pdfs_matching_criteria(criteria_data, idx)
                
                # Calculate incremental impact
                # If this criteria is already selected, don't show incremental count
                # If not selected, show how many new PDFs it would add
                
                label = f"{criteria_name} ({len(matching_pdfs)} PDFs"
                
                if idx not in st.session_state.selected_criteria:
                    # Only show incremental count if this criteria is NOT selected
                    new_pdfs = set(matching_pdfs) - current_selected
                    incremental = len(new_pdfs)
                    if incremental > 0 and len(current_selected) > 0:
                        label += f", +{incremental} new"
                
                label += ")"
                
                # Use session state to determine checkbox value
                checkbox_value = idx in st.session_state.selected_criteria
                
                # Check if checkbox state changed
                checkbox_result = st.checkbox(label, value=checkbox_value, key=f"criteria_{idx}")
                
                if checkbox_result != checkbox_value:
                    # State changed, update session state and rerun
                    if checkbox_result:
                        st.session_state.selected_criteria.add(idx)
                    else:
                        st.session_state.selected_criteria.discard(idx)
                    st.rerun()
                
                # Add to selected_indices if currently selected
                if checkbox_result:
                    if idx not in selected_indices:
                        selected_indices.append(idx)
        
        # Add selected criteria from session state to selected_indices
        for idx in st.session_state.selected_criteria:
            if idx not in selected_indices:
                selected_indices.append(idx)
    
    # Main content area
    # Calculate selected PDFs
    selected_pdfs = set()
    for idx in selected_indices:
        if 'sample size' in criteria_data[idx]['criteria'].lower():
            selected_pdfs.update(get_pdfs_matching_criteria(criteria_data, idx, sample_size_range))
        else:
            selected_pdfs.update(get_pdfs_matching_criteria(criteria_data, idx))
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total PDFs", total_pdfs)
    with col2:
        st.metric("Selected PDFs", len(selected_pdfs))
    with col3:
        st.metric("Coverage", f"{len(selected_pdfs)/total_pdfs*100:.1f}%")
    with col4:
        st.metric("Active Criteria", len(selected_indices))
    
    # Progress bar
    st.plotly_chart(create_progress_bar(len(selected_pdfs), total_pdfs), use_container_width=True)


    # Add Sankey chart below the progress bar
    if sankey_df is not None:
        sankey_fig = create_sankey_chart(sankey_df)
        if sankey_fig:
            st.plotly_chart(sankey_fig, use_container_width=True, config={
                'displayModeBar': True,
                'staticPlot': False,
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'sankey_chart',
                    'height': 600,
                    'width': 1200,
                    'scale': 1
                }
            })
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Incremental Impact", "Product Analysis", "Health Endpoints", "Selected PDFs"])
    
    with tab1:
        # Category distribution
        st.plotly_chart(create_category_distribution(criteria_data, selected_indices, pdf_columns, sample_size_range), use_container_width=True)
        
        # Criteria effectiveness
        if selected_indices:
            st.subheader("Criteria Effectiveness")
            criteria_effectiveness = []
            for idx in selected_indices:
                if 'sample size' in criteria_data[idx]['criteria'].lower():
                    matching_pdfs = get_pdfs_matching_criteria(criteria_data, idx, sample_size_range)
                else:
                    matching_pdfs = get_pdfs_matching_criteria(criteria_data, idx)
                
                criteria_effectiveness.append({
                    'Criteria': f"{criteria_data[idx]['category']} - {criteria_data[idx]['criteria']}",
                    'PDFs Matched': len(matching_pdfs),
                    'Percentage': f"{len(matching_pdfs)/total_pdfs*100:.1f}%"
                })
            
            df_effectiveness = pd.DataFrame(criteria_effectiveness)
            df_effectiveness = df_effectiveness.sort_values('PDFs Matched', ascending=False)
            st.dataframe(df_effectiveness, use_container_width=True, hide_index=True)
        
        # Raw data table (transposed) - UPDATED SECTION
        st.subheader("Screening Data")
        
        # Create both display and original data versions
        display_data, original_transposed_data = create_display_dataframe(screening_data)
        
        # Initialize session state for edited data if not exists
        if 'edited_data' not in st.session_state:
            st.session_state.edited_data = display_data.copy()
            st.session_state.original_data = original_transposed_data.copy()  # Keep original for restoration
            # Create a mapping from display index to original index
            st.session_state.index_mapping = {i: i for i in range(len(display_data))}
        
        # Refresh data when it's been updated externally (like after GitHub save)
        if not st.session_state.edited_data.equals(display_data):
            # Check if this is a real external change or just our own edits
            current_display_data, current_original_data = create_display_dataframe(screening_data)
            
            # If the underlying data has changed (external update), refresh session state
            if not st.session_state.original_data.equals(current_original_data):
                st.session_state.edited_data = current_display_data.copy()
                st.session_state.original_data = current_original_data.copy()
                st.session_state.index_mapping = {i: i for i in range(len(current_display_data))}
                st.info("📱 Data refreshed from external changes")
        
        # Find the rejection reasons column
        rejection_reasons_col = None
        for col in display_data.columns:
            if 'Rejection reasons' in col:
                rejection_reasons_col = col
                break
        
        # Create editing interface for rejection reasons
        if rejection_reasons_col:
            st.subheader("Edit Rejection Reasons")
            
            # Create expandable editor for each row
            with st.expander("📝 Click to edit rejection reasons for individual PDFs"):
                # Use a container with height constraint
                container = st.container(height=400)
                
                with container:
                    for display_idx, row in st.session_state.edited_data.iterrows():
                        # Get the original index from our mapping
                        original_idx = st.session_state.index_mapping[display_idx]
                        pdf_name = row['PDF']
                        current_reason = str(row[rejection_reasons_col]) if pd.notna(row[rejection_reasons_col]) else ""
                        
                        # Create a text input for each PDF - use PDF name in key to avoid conflicts
                        new_reason = st.text_input(
                            f"**{display_idx + 1}. {pdf_name}**",  # Show numbering in label
                            value=current_reason,
                            key=f"rejection_reason_{pdf_name}_{display_idx}",  # Use display index for uniqueness
                            help="Edit the rejection reason for this PDF"
                        )
                        
                        # Update both edited data and original data if changed
                        if new_reason != current_reason:
                            st.session_state.edited_data.loc[display_idx, rejection_reasons_col] = new_reason
                            # Update the original data using the mapped original index
                            st.session_state.original_data.loc[original_idx, rejection_reasons_col] = new_reason
        
        # Create styling functions
        def color_screening_classification(val):
            """Apply color coding to screening classification values"""
            if val == 'Needs Summary':
                return 'background-color: #d4edda'  # Pastel green
            elif val == 'Record for Later':
                return 'background-color: #fff3cd'  # Pastel yellow
            elif val == 'No Action':
                return 'background-color: #f8d7da'  # Pastel red
            return ''
        
        def color_quality_values(val):
            """Apply color coding to quality-related values"""
            val_str = str(val).lower()
            if 'high' in val_str or 'top-tier' in val_str:
                return 'background-color: #d4edda'  # Pastel green
            elif 'medium' in val_str or 'moderate' in val_str or 'mid-tier' in val_str:
                return 'background-color: #fff3cd'  # Pastel yellow
            elif 'low' in val_str or 'unknown' in val_str or 'low-tier' in val_str:
                return 'background-color: #f8d7da'  # Pastel red
            return ''
        
        def color_boolean_values(val):
            """Apply color coding to boolean values"""
            if str(val).lower() == 'true':
                return 'color: #28a745'  # Green text
            elif str(val).lower() == 'false':
                return 'color: #dc3545'  # Red text
            return ''
        
        # Find columns for styling
        screening_col = None
        quality_cols = []
        boolean_cols = []
        
        # List of columns that contain boolean values
        boolean_criteria = [
            'Competitor-funded research',
            'Priority health endpoints', 
            'Nicotine general studies',
            'Peer review status',
            'Novel/interesting findings'
        ]
        
        # List of quality-related criteria
        quality_criteria = [
            'Journal quality',
            'Methodology appropriateness',
            'Quality assessment'
        ]
        
        for col in st.session_state.edited_data.columns:
            if 'Screening classification' in col:
                screening_col = col
            
            # Check if this column contains quality criteria
            for criteria in quality_criteria:
                if criteria in col:
                    quality_cols.append(col)
                    break
            
            # Check if this column contains boolean criteria
            for criteria in boolean_criteria:
                if criteria in col:
                    boolean_cols.append(col)
                    break
        
        # Create a copy for display with proper numbering and set it as index
        display_df = st.session_state.edited_data.copy()
        
        # Set the index to start from 1 for display purposes
        display_df.index = range(1, len(display_df) + 1)
        display_df.index.name = 'No.'
        
        # Apply styling to the display dataframe
        styled_display_df = display_df.style
        
        # Apply screening classification colors
        if screening_col:
            styled_display_df = styled_display_df.map(
                color_screening_classification, 
                subset=[screening_col]
            )
        
        # Apply quality value colors
        if quality_cols:
            styled_display_df = styled_display_df.map(
                color_quality_values,
                subset=quality_cols
            )
        
        # Apply boolean value colors
        if boolean_cols:
            styled_display_df = styled_display_df.map(
                color_boolean_values,
                subset=boolean_cols
            )
        
        # Display the styled dataframe - this will show only the 'No.' index column
        st.dataframe(styled_display_df, use_container_width=True, height=400)
        
        # Show changes indicator - compare with fresh data
        baseline_display_data, _ = create_display_dataframe(screening_data)
        if not st.session_state.edited_data.equals(baseline_display_data):
            st.success("✅ Changes detected in rejection reasons!")
            
            # Add save button
            if st.button("💾 Save Changes", type="primary"):
                try:
                    # Use the original data with edits applied for saving
                    data_to_save = st.session_state.original_data.copy()
                    
                    # Save to GitHub using the original data structure
                    success = save_to_github_direct(data_to_save)
                    if success:
                        st.success("Changes saved to GitHub successfully!")
                        # Refresh the baseline data after successful save
                        fresh_display, fresh_original = create_display_dataframe(screening_data)
                        st.session_state.edited_data = fresh_display.copy()
                        st.session_state.original_data = fresh_original.copy()
                        st.session_state.index_mapping = {i: i for i in range(len(fresh_display))}
                        st.rerun()
                    else:
                        st.error("Failed to save changes to GitHub")
                except Exception as e:
                    st.error(f"Error saving changes: {e}")

        # Show what changed
        with st.expander("📋 View Changes"):
            changes = []
            baseline_display_data, _ = create_display_dataframe(screening_data)
            
            for display_idx in st.session_state.edited_data.index:
                if rejection_reasons_col:
                    original = str(baseline_display_data.loc[display_idx, rejection_reasons_col]) if pd.notna(baseline_display_data.loc[display_idx, rejection_reasons_col]) else ""
                    edited = str(st.session_state.edited_data.loc[display_idx, rejection_reasons_col]) if pd.notna(st.session_state.edited_data.loc[display_idx, rejection_reasons_col]) else ""
                    
                    if original != edited:
                        pdf_name = st.session_state.edited_data.loc[display_idx, 'PDF']
                        changes.append({
                            'No.': display_idx + 1,
                            'PDF': pdf_name,
                            'Original': original,
                            'New': edited
                        })
            
            if changes:
                changes_df = pd.DataFrame(changes)
                st.dataframe(changes_df, use_container_width=True, hide_index=True)
            else:
                st.info("No changes detected")
        
        # Reset button
        if st.button("🔄 Reset All Changes"):
            display_data, original_transposed_data = create_display_dataframe(screening_data)
            st.session_state.edited_data = display_data.copy()
            st.session_state.original_data = original_transposed_data.copy()
            st.session_state.index_mapping = {i: i for i in range(len(display_data))}
            st.rerun()
        
        # Add download button for the edited Excel data
        def convert_to_excel_with_formatting(df):
            """Convert dataframe to Excel with formatting intact"""
            from io import BytesIO
            import xlsxwriter
            
            output = BytesIO()
            
            # Create a workbook with nan_inf_to_errors option
            workbook = xlsxwriter.Workbook(output, {'in_memory': True, 'nan_inf_to_errors': True})
            worksheet = workbook.add_worksheet('Screening Data')
            
            # Define formats for different categories
            screening_formats = {
                'Needs Summary': workbook.add_format({'bg_color': '#d4edda'}),
                'Record for Later': workbook.add_format({'bg_color': '#fff3cd'}),
                'No Action': workbook.add_format({'bg_color': '#f8d7da'})
            }
            
            quality_formats = {
                'high': workbook.add_format({'bg_color': '#d4edda'}),
                'medium': workbook.add_format({'bg_color': '#fff3cd'}),
                'low': workbook.add_format({'bg_color': '#f8d7da'})
            }
            
            boolean_formats = {
                'True': workbook.add_format({'font_color': '#28a745'}),
                'False': workbook.add_format({'font_color': '#dc3545'})
            }
            
            # Create download dataframe with numbering for better user experience
            download_df = df.copy()
            download_df.index = range(1, len(download_df) + 1)
            download_df.index.name = 'No.'
            
            # Write the index header
            worksheet.write(0, 0, 'No.')
            
            # Write other column headers
            for col_idx, col_name in enumerate(download_df.columns, start=1):
                worksheet.write(0, col_idx, col_name)
            
            # Write data with formatting
            for row_idx, (index, row) in enumerate(download_df.iterrows(), start=1):
                # Write the index (row number)
                worksheet.write(row_idx, 0, index)
                
                # Write the data
                for col_idx, (col_name, value) in enumerate(row.items(), start=1):
                    cell_format = None
                    
                    # Handle NaN/None values
                    if pd.isna(value) or value is None:
                        value = ""
                    
                    # Check for screening classification formatting
                    if 'Screening classification' in col_name:
                        if value in screening_formats:
                            cell_format = screening_formats[value]
                    
                    # Check for quality formatting
                    elif any(criteria in col_name for criteria in ['Journal quality', 'Methodology appropriateness', 'Quality assessment']):
                        val_str = str(value).lower()
                        if 'high' in val_str or 'top-tier' in val_str:
                            cell_format = quality_formats['high']
                        elif 'medium' in val_str or 'moderate' in val_str or 'mid-tier' in val_str:
                            cell_format = quality_formats['medium']
                        elif 'low' in val_str or 'unknown' in val_str or 'low-tier' in val_str:
                            cell_format = quality_formats['low']
                    
                    # Check for boolean formatting
                    elif any(criteria in col_name for criteria in ['Competitor-funded research', 'Priority health endpoints', 'Nicotine general studies', 'Peer review status', 'Novel/interesting findings']):
                        if str(value).lower() == 'true':
                            cell_format = boolean_formats['True']
                        elif str(value).lower() == 'false':
                            cell_format = boolean_formats['False']
                    
                    # Write the cell with or without formatting
                    if cell_format:
                        worksheet.write(row_idx, col_idx, value, cell_format)
                    else:
                        worksheet.write(row_idx, col_idx, value)
            
            # Auto-adjust column widths
            worksheet.set_column(0, 0, 5)  # No. column
            for col_idx, col_name in enumerate(download_df.columns, start=1):
                max_length = max(len(str(col_name)), max(len(str(download_df.iloc[row_idx, col_idx-1])) for row_idx in range(len(download_df))))
                worksheet.set_column(col_idx, col_idx, min(max_length + 2, 50))
            
            workbook.close()
            output.seek(0)
            return output.getvalue()
        
        # Use the display data for Excel export (formatted for readability)
        excel_data = convert_to_excel_with_formatting(st.session_state.edited_data)
        st.download_button(
            label="📥 Download Data",
            data=excel_data,
            file_name="screening_data_edited.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    
    with tab2:
        st.subheader("Incremental Impact Analysis")
        if len(selected_indices) > 0:
            fig = create_incremental_impact_chart(criteria_data, selected_indices, pdf_columns, sample_size_range)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("All remaining criteria would add 0 new PDFs to your current selection")
        else:
            # Show initial impact of all criteria when nothing is selected
            # Use same format as the main chart for consistency
            
            # Color palette for categories (same as main chart)
            category_colors = {
                'Product Mentions': '#1f77b4',      # Blue
                'Funding Sources': '#ff7f0e',       # Orange
                'Health Endpoints': '#2ca02c',      # Green
                'Product Categories': '#d62728',    # Red
                'Study Quality': '#9467bd',         # Purple
                'Others': '#e377c2',                # Pink
            }
            
            labels = []
            pdf_counts = []
            colors = []
            
            # Add all criteria in original order (as they appear in the data)
            for idx, criteria in enumerate(criteria_data):
                category = criteria['category']
                criteria_name = criteria['criteria']
                
                # Get matching PDFs for this criteria
                if 'sample size' in criteria_name.lower():
                    matching_pdfs = get_pdfs_matching_criteria(criteria_data, idx, sample_size_range)
                else:
                    matching_pdfs = get_pdfs_matching_criteria(criteria_data, idx)
                
                labels.append(criteria_name)
                pdf_counts.append(len(matching_pdfs))
                
                # Get color for this category
                color = category_colors.get(category, '#7f7f7f')  # Default gray if category not found
                colors.append(color)
            
            # Create horizontal bar chart
            fig = go.Figure()
            
            # Add PDF count bars
            fig.add_trace(go.Bar(
                name='PDFs Available',
                y=labels,
                x=pdf_counts,
                orientation='h',
                marker=dict(
                    color=colors,
                    opacity=1.0,
                    line=dict(color='rgba(0,0,0,0.5)', width=1)
                ),
                text=[f'{v}' for v in pdf_counts],
                textposition='inside',
                textfont=dict(color='white', size=10, weight='bold'),
                hovertemplate='%{y}<br>Available PDFs: %{x}<extra></extra>'
            ))
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text="PDF Coverage by Individual Criteria (No Selection)",
                    font=dict(size=18, weight='bold'),
                    x=0.5,
                    xanchor='center'
                ),
                xaxis=dict(
                    title="Number of PDFs",
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)',
                    range=[0, max(pdf_counts) * 1.1] if pdf_counts else [0, 1]
                ),
                yaxis=dict(
                    title="",
                    autorange='reversed',  # Show from top to bottom (same as main chart)
                    tickfont=dict(size=11)
                ),
                height=max(400, len(labels) * 35),  # Dynamic height: minimum 400px, 35px per bar
                margin=dict(l=300, r=100, t=80, b=50),  # Larger left margin for labels
                showlegend=False,
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)

    
    with tab3:
        st.subheader("Product Analysis")
        product_figs = create_product_distribution(criteria_data, selected_indices)
        if product_figs:
            for fig in product_figs:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No product data found in the dataset")
    
    with tab4:
            st.subheader("Health Endpoints Distribution")
            
            # Create visualization for health endpoints - shows all data regardless of selection
            health_conditions = defaultdict(int)
            
            # Process ALL health conditions data (similar to imperial products logic)
            for idx, criteria in enumerate(criteria_data):
                if criteria['criteria'] == 'Specific health conditions':
                    for i, value in enumerate(criteria['values']):
                        # Skip completely empty values
                        if pd.isna(value):
                            continue
                        
                        parsed = parse_value(value)
                        
                        # Handle list of health conditions (this is the main case we expect)
                        if isinstance(parsed, list) and len(parsed) > 0:
                            for condition in parsed:
                                if condition and str(condition).strip():
                                    clean_condition = str(condition).strip()
                                    # Remove any brackets or quotes that might be present
                                    clean_condition = clean_condition.replace('[', '').replace(']', '').replace('"', '').replace("'", "")
                                    if clean_condition:  # Only add non-empty conditions
                                        health_conditions[clean_condition] += 1
                        
                        # Handle single condition as string (fallback case)
                        elif isinstance(parsed, str) and parsed.strip() and parsed.strip() != '[]':
                            clean_condition = parsed.strip()
                            if clean_condition not in ['Unknown', 'No Action', '[]', '']:
                                # If it looks like a list string, try to parse it further
                                if clean_condition.startswith('[') and clean_condition.endswith(']'):
                                    # Remove brackets and split by comma
                                    inner_content = clean_condition[1:-1]
                                    if inner_content.strip():
                                        conditions = [c.strip().replace('"', '').replace("'", "") for c in inner_content.split(',')]
                                        for condition in conditions:
                                            if condition and condition.strip():
                                                health_conditions[condition.strip()] += 1
                                else:
                                    health_conditions[clean_condition] += 1
            
            if health_conditions:
                # Create horizontal bar chart for better readability
                conditions_list = list(health_conditions.keys())
                counts_list = list(health_conditions.values())
                
                # Sort by count (descending) for better visualization
                sorted_data = sorted(zip(conditions_list, counts_list), key=lambda x: x[1], reverse=True)
                sorted_conditions, sorted_counts = zip(*sorted_data)
                
                fig = go.Figure()
                
                # Create horizontal bar chart with health-themed color
                fig.add_trace(go.Bar(
                    name='Health Conditions',
                    y=list(sorted_conditions),
                    x=list(sorted_counts),
                    orientation='h',
                    marker=dict(
                        color="#6be399",  # Green color for health theme
                        opacity=1.0,
                        line=dict(color='rgba(0,0,0,0.3)', width=1)
                    ),
                    text=[f'{count}' for count in sorted_counts],
                    textposition='inside',
                    textfont=dict(color='white', size=10, weight='bold'),
                    hovertemplate='%{y}<br>Papers: %{x}<extra></extra>'
                ))
                
                # Update layout for better readability
                fig.update_layout(
                    title=dict(
                        text="Health Conditions Distribution Across Papers",
                        font=dict(size=18, weight='bold'),
                        x=0.5,
                        xanchor='center'
                    ),
                    xaxis=dict(
                        title="Number of Papers",
                        showgrid=True,
                        gridcolor='rgba(0,0,0,0.1)',
                        range=[0, max(sorted_counts) * 1.1] if sorted_counts else [0, 1]
                    ),
                    yaxis=dict(
                        title="Health Conditions",
                        autorange='reversed',  # Show highest counts at top
                        tickfont=dict(size=11)
                    ),
                    height=max(400, len(conditions_list) * 35),  # Dynamic height based on number of conditions
                    margin=dict(l=200, r=100, t=80, b=50),  # Larger left margin for condition names
                    showlegend=False,
                    plot_bgcolor='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)

    
    with tab5:
        st.subheader("Selected PDFs Based on Chosen Criterias")
        if selected_pdfs:
            # Create a detailed view of selected PDFs
            pdf_details = []
            for pdf in selected_pdfs:
                pdf_criteria = []
                for idx in selected_indices:
                    criteria = criteria_data[idx]
                    pdf_idx = criteria['pdf_columns'].index(pdf)
                    value = parse_value(criteria['values'][pdf_idx])
                    
                    # Check if this PDF matches this criteria
                    matches = False
                    
                    # Handle sample size criteria
                    if 'sample size' in criteria['criteria'].lower():
                        sample_size = None
                        if isinstance(value, (int, float)) and not pd.isna(value):
                            sample_size = int(value)
                        elif isinstance(value, str):
                            import re
                            numbers = re.findall(r'\d+', str(value))
                            if numbers:
                                try:
                                    sample_size = int(numbers[0])
                                except:
                                    pass
                        
                        if sample_size is not None and sample_size_range and sample_size_range[0] <= sample_size <= sample_size_range[1]:
                            matches = True
                    else:
                        # List of boolean criteria (without trailing spaces)
                        boolean_criteria = [
                            'Competitor-funded research',
                            'Priority health endpoints',
                            'Nicotine general studies',
                            'Peer review status',
                            'Novel/interesting findings'
                        ]
                        
                        criteria_name = criteria['criteria']
                        
                        # For boolean criteria, only consider True values
                        if criteria_name in boolean_criteria:
                            if value is True:
                                matches = True
                        # For other criteria types
                        elif isinstance(value, bool) and value:
                            matches = True
                        elif isinstance(value, list) and len(value) > 0:
                            matches = True
                        elif isinstance(value, dict) and len(value) > 0:
                            matches = True
                        elif isinstance(value, str) and value not in ['', 'Unknown', 'No Action', 'False', 'false']:
                            matches = True
                    
                    if matches:
                        pdf_criteria.append(f"{criteria['category']} - {criteria['criteria']}")
                
                pdf_details.append({
                    'PDF': pdf,
                    'Matching Criteria': len(pdf_criteria),
                    'Criteria Details': ', '.join(pdf_criteria[:3]) + ('...' if len(pdf_criteria) > 3 else '')
                })
            
            df_pdfs = pd.DataFrame(pdf_details)
            df_pdfs = df_pdfs.sort_values('Matching Criteria', ascending=False)
            
            # Display summary
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Selected PDFs", len(selected_pdfs))
            with col2:
                avg_criteria = df_pdfs['Matching Criteria'].mean()
                st.metric("Avg Criteria per PDF", f"{avg_criteria:.1f}")
            
            # Display table
            st.dataframe(df_pdfs, use_container_width=True, hide_index=True)
            
            # Download button
            csv = df_pdfs.to_csv(index=False)
            st.download_button(
                label="Download Selected PDFs List",
                data=csv,
                file_name="selected_pdfs.csv",
                mime="text/csv"
            )
        else:
            st.info("No PDFs selected. Please select screening criteria from the sidebar.")


if __name__ == "__main__":
    main()
