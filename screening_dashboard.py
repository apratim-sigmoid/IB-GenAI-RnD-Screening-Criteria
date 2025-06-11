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
        

@st.cache_data
def load_data():
    """Load and process the screening criteria dataset"""
    # Read the Excel file as screening_data (includes metadata)
    screening_data = pd.read_excel('screening_criteria_dataset.xlsx')
    
    # Create df by excluding metadata category
    df = screening_data[screening_data['Category'] != 'Metadata'].copy()
    
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
    
    return df, criteria_data, pdf_columns

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
                if criteria['criteria'] == 'Screening classification' and parsed_value in ['Needs Summary', 'Record for Later']:
                    matching_pdfs.append(pdf_name)
                elif criteria['criteria'] != 'Screening classification':
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
        'Classification System': '#8c564b', # Brown
        'Others': '#e377c2',                # Pink
        'Other': '#e377c2'                  # Pink (alternate spelling)
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
    """Create a chart showing incremental impact of each criteria"""
    current_pdfs = set()
    for idx in selected_indices:
        if 'sample size' in criteria_data[idx]['criteria'].lower():
            current_pdfs.update(get_pdfs_matching_criteria(criteria_data, idx, sample_size_range))
        else:
            current_pdfs.update(get_pdfs_matching_criteria(criteria_data, idx))
    
    impact_data = []
    for idx, criteria in enumerate(criteria_data):
        if idx not in selected_indices:
            if 'sample size' in criteria['criteria'].lower():
                new_pdfs = set(get_pdfs_matching_criteria(criteria_data, idx, sample_size_range))
            else:
                new_pdfs = set(get_pdfs_matching_criteria(criteria_data, idx))
            
            incremental = len(new_pdfs - current_pdfs)
            total = len(new_pdfs)
            overlap = len(new_pdfs & current_pdfs)
            
            impact_data.append({
                'criteria': f"{criteria['category']} - {criteria['criteria']}",
                'incremental': incremental,
                'overlap': overlap,
                'total': total
            })
    
    # Sort by incremental impact
    impact_data.sort(key=lambda x: x['incremental'], reverse=True)
    impact_data = impact_data[:10]  # Top 10
    
    if impact_data:
        df_impact = pd.DataFrame(impact_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='New PDFs',
            x=df_impact['criteria'],
            y=df_impact['incremental'],
            marker_color='green'
        ))
        fig.add_trace(go.Bar(
            name='Already Selected',
            x=df_impact['criteria'],
            y=df_impact['overlap'],
            marker_color='gray'
        ))
        
        fig.update_layout(
            title="Top 10 Criteria by Incremental Impact",
            xaxis_title="Criteria",
            yaxis_title="Number of PDFs",
            barmode='stack',
            height=500,
            xaxis_tickangle=-45
        )
        
        return fig
    return None

def create_product_distribution(criteria_data, selected_indices):
    """Create visualization for product mentions"""
    imperial_products = defaultdict(int)
    competitor_products = defaultdict(lambda: defaultdict(int))
    
    # Find product mention criteria
    for idx, criteria in enumerate(criteria_data):
        if idx in selected_indices:
            if criteria['criteria'] == 'Imperial Products (e.g. Blu)':
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
                'Philip Morris International': '#d62728',  # Red
                'JTI': '#2ca02c',             # Green
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
    st.markdown("Select screening criteria to filter research papers")
    
    # Load data
    df, criteria_data, pdf_columns = load_data()
    total_pdfs = len(pdf_columns)
    
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
                current_selected = set()
                for sel_idx in selected_indices:
                    if 'sample size' in criteria_data[sel_idx]['criteria'].lower():
                        current_selected.update(get_pdfs_matching_criteria(criteria_data, sel_idx, sample_size_range))
                    else:
                        current_selected.update(get_pdfs_matching_criteria(criteria_data, sel_idx))
                
                new_pdfs = set(matching_pdfs) - current_selected
                incremental = len(new_pdfs)
                
                label = f"{criteria_name} ({len(matching_pdfs)} PDFs"
                if incremental > 0 and len(current_selected) > 0:
                    label += f", +{incremental} new"
                label += ")"
                
                # Use session state to determine checkbox value
                checkbox_value = idx in st.session_state.selected_criteria
                
                if st.checkbox(label, value=checkbox_value, key=f"criteria_{idx}"):
                    st.session_state.selected_criteria.add(idx)
                    if idx not in selected_indices:
                        selected_indices.append(idx)
                else:
                    st.session_state.selected_criteria.discard(idx)
        
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
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Incremental Impact", "Product Analysis", "Selected PDFs"])
    
    with tab1:
        # Helper function for formatting competitor products
        def format_competitor_products(value):
            """Format competitor products dictionary into readable format"""
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
        
        # Raw data table (transposed)
        st.subheader("Screening Data")
        
        # Read the full screening data including metadata
        screening_data = pd.read_excel('screening_criteria_dataset.xlsx')
        
        # Transpose the data so criteria are columns and PDFs are rows
        transposed_data = screening_data.set_index(['Category', 'Specific Criteria']).T
        
        # Create multi-level column headers
        transposed_data.columns = [f"{cat} - {crit}" for cat, crit in transposed_data.columns]
        
        # Reset index to make PDF names a column
        transposed_data = transposed_data.reset_index()
        transposed_data = transposed_data.rename(columns={'index': 'PDF'})
        
        # Clean up list formatting - remove square brackets
        for col in transposed_data.columns:
            if col != 'PDF':  # Don't modify the PDF column
                if 'Competitor Products' in col:
                    # Special handling for competitor products dictionary format
                    transposed_data[col] = transposed_data[col].apply(lambda x: format_competitor_products(x))
                else:
                    # General formatting for other columns
                    transposed_data[col] = transposed_data[col].apply(lambda x: 
                        str(x).replace('[', '').replace(']', '').replace("'", '') if pd.notna(x) and str(x) not in ['{}', ''] else 
                        '' if str(x) == '{}' else x)
        
        # Reorder columns to move Classification System columns next to PDF
        all_columns = list(transposed_data.columns)
        pdf_col = ['PDF']
        classification_cols = [col for col in all_columns if 'Classification System' in col]
        other_cols = [col for col in all_columns if col != 'PDF' and 'Classification System' not in col]
        
        # New column order: PDF, Classification System columns, then all others
        new_column_order = pdf_col + classification_cols + other_cols
        transposed_data = transposed_data[new_column_order]
        
        # Add sequential numbering starting from 1 and set as index
        transposed_data.index = range(1, len(transposed_data) + 1)
        transposed_data.index.name = 'No.'
        
        # Initialize session state for edited data if not exists
        if 'edited_data' not in st.session_state:
            st.session_state.edited_data = transposed_data.copy()
        
        # Find the rejection reasons column
        rejection_reasons_col = None
        for col in transposed_data.columns:
            if 'Rejection reasons' in col:
                rejection_reasons_col = col
                break
        
        # Create editing interface for rejection reasons
        if rejection_reasons_col:
            st.subheader("Edit Rejection Reasons")
            
            # Create expandable editor for each row
            with st.expander("📝 Click to edit rejection reasons for individual PDFs"):
                for idx, row in st.session_state.edited_data.iterrows():
                    pdf_name = row['PDF']
                    current_reason = str(row[rejection_reasons_col]) if pd.notna(row[rejection_reasons_col]) else ""
                    
                    # Create a text input for each PDF
                    new_reason = st.text_area(
                        f"**{pdf_name}**",
                        value=current_reason,
                        height=68,
                        key=f"rejection_reason_{idx}",
                        help="Edit the rejection reason for this PDF"
                    )
                    
                    # Update the edited data if changed
                    if new_reason != current_reason:
                        st.session_state.edited_data.loc[idx, rejection_reasons_col] = new_reason
        
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
        
        # Apply styling to the edited data
        styled_df = st.session_state.edited_data.style
        
        # Apply screening classification colors
        if screening_col:
            styled_df = styled_df.applymap(
                color_screening_classification, 
                subset=[screening_col]
            )
        
        # Apply quality value colors
        if quality_cols:
            styled_df = styled_df.applymap(
                color_quality_values,
                subset=quality_cols
            )
        
        # Apply boolean value colors
        if boolean_cols:
            styled_df = styled_df.applymap(
                color_boolean_values,
                subset=boolean_cols
            )
        
        # Display the styled dataframe
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Show changes indicator
        if not st.session_state.edited_data.equals(transposed_data):
            st.success("✅ Changes detected in rejection reasons!")
            
            # Add save button
            if st.button("💾 Save Changes", type="primary"):
                try:
                    # Save to GitHub using direct push
                    success = save_to_github_direct(st.session_state.edited_data)
                    if success:
                        st.success("Changes saved to GitHub successfully!")
                    else:
                        st.error("Failed to save changes to GitHub")
                except Exception as e:
                    st.error(f"Error saving changes: {e}")


    def convert_back_to_original_format(transposed_df):
        """Convert the transposed dataframe back to original Excel format"""
        # This function needs to reverse the transposition you did earlier
        # You'll need to implement this based on your exact data structure
        
        # Remove the sequential numbering and get back to original structure
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
        
        return original_df
            
        # Show what changed
        with st.expander("📋 View Changes"):
            changes = []
            for idx in st.session_state.edited_data.index:
                original = str(transposed_data.loc[idx, rejection_reasons_col]) if pd.notna(transposed_data.loc[idx, rejection_reasons_col]) else ""
                edited = str(st.session_state.edited_data.loc[idx, rejection_reasons_col]) if pd.notna(st.session_state.edited_data.loc[idx, rejection_reasons_col]) else ""
                
                if original != edited:
                    pdf_name = st.session_state.edited_data.loc[idx, 'PDF']
                    changes.append({
                        'PDF': pdf_name,
                        'Original': original,
                        'New': edited
                    })
            
            if changes:
                changes_df = pd.DataFrame(changes)
                st.dataframe(changes_df, use_container_width=True, hide_index=True)
        
        # Reset button
        if st.button("🔄 Reset All Changes"):
            st.session_state.edited_data = transposed_data.copy()
            st.rerun()
        
        # Add download button for the edited Excel data
        @st.cache_data
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
            
            # Write headers
            for col_idx, col_name in enumerate(df.columns):
                worksheet.write(0, col_idx, col_name)
            
            # Write data with formatting
            for row_idx, (index, row) in enumerate(df.iterrows(), start=1):
                for col_idx, (col_name, value) in enumerate(row.items()):
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
            for col_idx, col_name in enumerate(df.columns):
                max_length = max(len(str(col_name)), max(len(str(df.iloc[row_idx, col_idx])) for row_idx in range(len(df))))
                worksheet.set_column(col_idx, col_idx, min(max_length + 2, 50))
            
            workbook.close()
            output.seek(0)
            return output.getvalue()
        
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
                st.info("Select some criteria to see incremental impact of additional criteria")
        else:
            # Show initial impact of all criteria
            impact_data = []
            for idx, criteria in enumerate(criteria_data):
                if 'sample size' in criteria['criteria'].lower():
                    matching_pdfs = get_pdfs_matching_criteria(criteria_data, idx, sample_size_range)
                else:
                    matching_pdfs = get_pdfs_matching_criteria(criteria_data, idx)
                
                impact_data.append({
                    'Criteria': f"{criteria['category']} - {criteria['criteria']}",
                    'PDFs': len(matching_pdfs)
                })
            
            df_impact = pd.DataFrame(impact_data).sort_values('PDFs', ascending=False).head(15)
            fig = px.bar(df_impact, x='Criteria', y='PDFs', title="Top 15 Criteria by PDF Coverage")
            fig.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Product Analysis")
        product_figs = create_product_distribution(criteria_data, selected_indices)
        if product_figs:
            for fig in product_figs:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select product-related criteria to see product distribution analysis")
        
        # Health endpoints analysis
        if any(criteria_data[idx]['criteria'] == 'Specific health conditions' for idx in selected_indices):
            st.subheader("Health Endpoints Distribution")
            health_conditions = defaultdict(int)
            
            for idx in selected_indices:
                if criteria_data[idx]['criteria'] == 'Specific health conditions':
                    for value in criteria_data[idx]['values']:
                        parsed = parse_value(value)
                        if isinstance(parsed, list):
                            for condition in parsed:
                                health_conditions[condition] += 1
            
            if health_conditions:
                fig = px.bar(
                    x=list(health_conditions.keys()),
                    y=list(health_conditions.values()),
                    title="Health Conditions Mentioned in Selected Papers",
                    labels={'x': 'Health Condition', 'y': 'Number of Papers'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Selected PDFs")
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
                            if criteria_name == 'Screening classification' and value in ['Needs Summary', 'Record for Later']:
                                matches = True
                            elif criteria_name != 'Screening classification':
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
