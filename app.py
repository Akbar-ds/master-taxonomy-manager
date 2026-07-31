import streamlit as st
from supabase import create_client, Client
import pandas as pd
from datetime import datetime, timedelta, date
from google import genai

# Initialize Gemini client using secrets
@st.cache_resource
def get_gemini_client():
  return genai.Client(api_key=st.secrets["gemini"]["api_key"])


def classify_batch_articles(df_articles, taxonomy_df):
  client = get_gemini_client()
  taxonomy_reference = taxonomy_df.to_string(index=False)

  results = []
  progress_bar = st.progress(0)
  total_rows = len(df_articles)

  for idx, row in df_articles.iterrows():
    # Adjust 'article_text' to match the column name in your uploaded file
    article_content = str(row.get("article_text", ""))

    prompt = f"""
        You are an expert media analyst and taxonomy classification engine.
        Analyze the following article content and categorize it strictly using ONLY the valid Categories, Subcategories, Topics, and Tonality rules provided in the Master Taxonomy Reference below.

        ### Master Taxonomy Reference:
        {taxonomy_reference}

        ### Strict Rules:
        1. Choose the Category, Subcategory, Topic, and Tonality strictly from the reference list.
        2. If the article does not fit any topic in the reference list, output Category as "Cannot analyze as no relevant topic found", and leave Subcategory and Topic as "N/A".
        3. Provide an "Overall Tonality" column indicating the general tone (Positive, Negative, Neutral, Mixed).

        ### Article Content:
        {article_content}

        Return your response strictly in this format:
        Category: [Value]
        Subcategory: [Value]
        Topic: [Value]
        Tonality: [Value]
        Overall Tonality: [Value]
        """

    try:
      response = client.models.generate_content(
          model="gemini-2.5-flash",
          contents=prompt,
      )
      text = response.text
      # Simple parser to extract fields from the text response
      parsed = {}
      for line in text.split("\n"):
        if ":" in line:
          parts = line.split(":", 1)
          parsed[parts[0].strip()] = parts[1].strip()

      results.append({
          "Category": parsed.get("Category", "Error"),
          "Subcategory": parsed.get("Subcategory", "N/A"),
          "Topic": parsed.get("Topic", "N/A"),
          "Tonality": parsed.get("Tonality", "N/A"),
          "Overall Tonality": parsed.get("Overall Tonality", "N/A"),
      })
    except Exception as e:
      results.append({
          "Category": "Error",
          "Subcategory": str(e),
          "Topic": "N/A",
          "Tonality": "N/A",
          "Overall Tonality": "N/A",
      })

    progress_bar.progress((idx + 1) / total_rows)

  return pd.concat(
      [df_articles.reset_index(drop=True), pd.DataFrame(results)], axis=1
  )


# --- Streamlit UI Component for Your App ---
st.setHeader = st.subheader
st.setHeader("🤖 Bulk Article Classification Pipeline")

uploaded_file = st.file_uploader(
    "Upload Excel or CSV containing your articles", type=["xlsx", "csv"]
)

if uploaded_file:
  if uploaded_file.name.endswith(".csv"):
    df_input = pd.read_csv(uploaded_file)
  else:
    df_input = pd.read_excel(uploaded_file)

  st.write("Preview of Uploaded Data:", df_input.head())

  # Assuming you fetch your taxonomy dataframe from Supabase or load it locally:
  # taxonomy_df = fetch_master_taxonomy_from_supabase()
  # For testing, you can pass your loaded taxonomy dataframe here.

  if st.button("🚀 Run AI Classification"):
    # Placeholder for taxonomy dataframe - ensure your taxonomy DataFrame is loaded here
    # e.g., taxonomy_df = pd.read_csv("taxonomy.csv") or fetched from Supabase
    st.info(
        "Processing articles through Gemini using your master taxonomy rules..."
    )
    # output_df = classify_batch_articles(df_input, taxonomy_df)
    # st.success("Classification complete!")
    # st.dataframe(output_df)





# Page Configuration
st.set_page_config(
    page_title="Master Taxonomy Classification Lookup & Manager",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS Styling & Attractive Color Palettes
st.markdown("""
    <style>
    /* Main Background with a soft gradient look */
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #f8fafc 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    h1 {
        color: #0f172a;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    /* Container/Card Styling with subtle shadows */
    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.03), 0 4px 6px -2px rgba(0, 0, 0, 0.02);
        transition: all 0.3s ease;
    }
    
    /* Button Micro-animations & Aesthetics */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border-radius: 10px;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.25s ease-in-out;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.2);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(29, 78, 216, 0.3);
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
    }
    </style>
""", unsafe_allow_html=True)

# 1. Initialize Supabase Connection
@st.cache_resource
def init_connection() -> Client:
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase = init_connection()

# Allowed options for dropdown validations
ALLOWED_TONALITY_OPTIONS = [
    "All Tonalities",
    "Only Positive",
    "Only Negative",
    "Only Neutral",
    "Neutral & Negative"
]

# Fetch current data from Supabase
def fetch_master_data():
    try:
        response = supabase.table("taxonomy_entries").select("*").order("created_at", desc=True).execute()
        return response.data
    except Exception as e:
        st.error(f"Error fetching data from Supabase: {e}")
        return []

data = fetch_master_data()

# Extract unique values for search filters and form options
categories = sorted(list(set([item.get("category") for item in data if item.get("category")])))
subcategories = sorted(list(set([item.get("subcategory") for item in data if item.get("subcategory")])))
topics = sorted(list(set([item.get("topic") for item in data if item.get("topic")])))

# Define the pop-up modal dialog for adding new entries
@st.dialog("📝 Add New Taxonomy Entry")
def add_taxonomy_modal(categories, subcategories, ALLOWED_TONALITY_OPTIONS):
    st.markdown("<p style='font-size: 13px; color: #64748b;'>Complete the form below to register a new master taxonomy configuration.</p>", unsafe_allow_html=True)
    st.warning("⚠️ **Authorization Notice:** Alert First Consult to your TL before making changes.")
    
    with st.form("new_entry_form", clear_on_submit=True):
        op_name = st.text_input("Your Name / User ID *", placeholder="Enter your full name...")
        
        form_category = st.selectbox("Category *", options=categories if categories else ["Interviews", "Grievances", "Projects and Infra", "Policies", "Analysis Report", "Rules Violation", "Technology", "Irregularities"])
        form_subcategory = st.selectbox("Subcategory (Optional)", options=[""] + subcategories)
        form_topic = st.text_input("Topic *", placeholder="Enter topic name...")
        form_tonality = st.selectbox("Tonality / Condition Rule *", options=ALLOWED_TONALITY_OPTIONS)
        
        submit_col, cancel_col = st.columns(2)
        with submit_col:
            submitted = st.form_submit_button("🚀 Submit & Sync", type="primary", use_container_width=True)
        with cancel_col:
            cancelled = st.form_submit_button("❌ Cancel", use_container_width=True)
            
        if submitted:
            cleaned_topic = form_topic.strip()
            cleaned_name = op_name.strip()
            if not cleaned_name or not cleaned_topic:
                st.error("Please fill out your Name and the Topic field.")
            else:
                try:
                    current_db_data = fetch_master_data()
                    
                    norm_category = form_category.strip().lower()
                    norm_subcategory = (form_subcategory.strip().lower() if form_subcategory and form_subcategory != "" else "")
                    norm_topic = cleaned_topic.lower()
                    norm_tonality = form_tonality.strip().lower()
                    
                    is_duplicate = False
                    for entry in current_db_data:
                        db_cat = (entry.get("category") or "").strip().lower()
                        db_subcat = (entry.get("subcategory") or "").strip().lower()
                        db_top = (entry.get("topic") or "").strip().lower()
                        db_ton = (entry.get("tonality") or "").strip().lower()
                        
                        if (db_cat == norm_category and 
                            db_subcat == norm_subcategory and 
                            db_top == norm_topic and 
                            db_ton == norm_tonality):
                            is_duplicate = True
                            break
                    
                    if is_duplicate:
                        st.error(f"❌ **Duplicate Error:** '{cleaned_topic}' under this exact category structure already exists.")
                    else:
                        supabase.table("taxonomy_entries").insert({
                            "category": form_category,
                            "subcategory": form_subcategory if form_subcategory != "" else None,
                            "topic": cleaned_topic,
                            "tonality": form_tonality,
                            "submitted_by": cleaned_name
                        }).execute()
                        
                        st.success("✨ Entry successfully added and synced!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to add entry: {e}")
        
        if cancelled:
            st.rerun()

# Define the pop-up modal dialog for editing an existing row entry
@st.dialog("✏️ Edit Taxonomy Entry")
def edit_taxonomy_modal(item, categories, subcategories, ALLOWED_TONALITY_OPTIONS):
    st.markdown("<p style='font-size: 13px; color: #64748b;'>Modify the taxonomy details below. Changes update the database timestamp and record modifier.</p>", unsafe_allow_html=True)
    st.warning("⚠️ **Confirmation Required:** Are you sure you want to make this change?")
    
    with st.form("edit_entry_form", clear_on_submit=False):
        editor_name = st.text_input("Your Name / User ID (Editor) *", placeholder="Enter your full name...")
        
        default_cat_idx = categories.index(item.get("category")) if item.get("category") in categories else 0
        edit_category = st.selectbox("Category *", options=categories, index=default_cat_idx)
        
        sub_opts = [""] + subcategories
        default_sub_idx = sub_opts.index(item.get("subcategory")) if item.get("subcategory") in sub_opts else 0
        edit_subcategory = st.selectbox("Subcategory (Optional)", options=sub_opts, index=default_sub_idx)
        
        edit_topic = st.text_input("Topic *", value=item.get("topic", ""))
        
        default_ton_idx = ALLOWED_TONALITY_OPTIONS.index(item.get("tonality")) if item.get("tonality") in ALLOWED_TONALITY_OPTIONS else 0
        edit_tonality = st.selectbox("Tonality / Condition Rule *", options=ALLOWED_TONALITY_OPTIONS, index=default_ton_idx)
        
        proceed_col, cancel_col = st.columns(2)
        with proceed_col:
            proceed_button = st.form_submit_button("✅ Proceed & Update", type="primary", use_container_width=True)
        with cancel_col:
            cancel_button = st.form_submit_button("❌ Cancel", use_container_width=True)
            
        if proceed_button:
            cleaned_topic = edit_topic.strip()
            cleaned_name = editor_name.strip()
            if not cleaned_name or not cleaned_topic:
                st.error("Please fill out your Name and the Topic field.")
            else:
                try:
                    update_payload = {
                        "category": edit_category,
                        "subcategory": edit_subcategory if edit_subcategory != "" else None,
                        "topic": cleaned_topic,
                        "tonality": edit_tonality,
                        "submitted_by": cleaned_name,
                        "created_at": datetime.utcnow().isoformat()
                    }
                    
                    supabase.table("taxonomy_entries").update(update_payload).eq("id", item.get("id")).execute()
                    
                    st.success("Edited in the database successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to update entry in database: {e}")
                    
        if cancel_button:
            st.rerun()

# Header Layout
st.markdown("<h1>🌍 Master Taxonomy Manager</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 16px; color: #475569;'>Seamlessly search, filter, analyze, and manage shared classification rules.</p>", unsafe_allow_html=True)
st.divider()

# Top Action Buttons Section
col_btn1, col_btn2, _ = st.columns([1, 1, 4])

with col_btn1:
    if st.button("➕ Add New Entry", use_container_width=True):
        add_taxonomy_modal(categories, subcategories, ALLOWED_TONALITY_OPTIONS)

with col_btn2:
    if data:
        df_export = pd.DataFrame(data)
        cols_to_drop = [col for col in ['id', 'created_at'] if col in df_export.columns]
        df_export = df_export.drop(columns=cols_to_drop)
        
        csv_data = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Master CSV",
            data=csv_data,
            file_name="Master_Taxonomy_Report.csv",
            mime="text/csv",
            use_container_width=True
        )

# Search Filtering Section Container
st.markdown("### 🔍 Search Database Filters")
with st.container():
    s_col1, s_col2, s_col3 = st.columns(3)

    with s_col1:
        search_topic = st.selectbox("Topic Filter", options=["All"] + topics)
    with s_col2:
        search_category = st.selectbox("Category Filter", options=["All"] + categories)
    with s_col3:
        search_subcategory = st.selectbox("Subcategory Filter", options=["All"] + subcategories)

    # Date Filter Options Selector
    f_col1, f_col2 = st.columns([2, 2])
    with f_col1:
        date_filter_option = st.selectbox(
            "📅 Filter by Creation/Update Date", 
            options=["All Time", "Today", "Two Weeks Old (Last 14 Days)", "One Month Old (Last 30 Days)", "Custom Range"]
        )
    
    custom_date_range = None
    if date_filter_option == "Custom Range":
        with f_col2:
            custom_date_range = st.date_input(
                "Select Custom Date Range",
                value=(date.today() - timedelta(days=7), date.today())
            )

# Filter logic processing
filtered_data = data
if search_topic != "All":
    filtered_data = [item for item in filtered_data if item.get("topic") == search_topic]
if search_category != "All":
    filtered_data = [item for item in filtered_data if item.get("category") == search_category]
if search_subcategory != "All":
    filtered_data = [item for item in filtered_data if item.get("subcategory") == search_subcategory]

# Apply Date Filtering Based on User Selection
if date_filter_option != "All Time":
    now_utc = datetime.utcnow()
    filtered_by_date = []
    
    for item in filtered_data:
        created_at_str = item.get("created_at")
        if not created_at_str:
            continue
        try:
            created_dt = datetime.fromisoformat(created_at_str.replace("Z", "+00:00").split("+")[0])
            item_date = created_dt.date()
            
            if date_filter_option == "Today":
                if item_date == now_utc.date():
                    filtered_by_date.append(item)
            elif date_filter_option == "Two Weeks Old (Last 14 Days)":
                if now_utc - created_dt <= timedelta(days=14):
                    filtered_by_date.append(item)
            elif date_filter_option == "One Month Old (Last 30 Days)":
                if now_utc - created_dt <= timedelta(days=30):
                    filtered_by_date.append(item)
            elif date_filter_option == "Custom Range" and custom_date_range:
                if len(custom_date_range) == 2:
                    start_d, end_d = custom_date_range
                    if start_d <= item_date <= end_d:
                        filtered_by_date.append(item)
                elif len(custom_date_range) == 1:
                    if item_date == custom_date_range[0]:
                        filtered_by_date.append(item)
        except Exception:
            pass
            
    filtered_data = filtered_by_date

st.divider()

# Results Summary Header with Refined Styling Pill
res_col1, res_col2 = st.columns([4, 1])
with res_col1:
    st.subheader("📋 Filtered Taxonomy Records")
with res_col2:
    st.markdown(f"<div style='text-align: right; background-color: #e0f2fe; color: #0369a1; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 13px; box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);'>📊 {len(filtered_data)} entries found</div>", unsafe_allow_html=True)

# Render Table Results with Clear Header Column Names and Action Button
if not filtered_data:
    st.info("ℹ️ No matching taxonomy records found matching your selected criteria.")
else:
    # Render table header names cleanly
    header_cols = st.columns([2, 2, 2, 2, 1.5, 1])
    with header_cols[0]:
        st.markdown("**Category**")
    with header_cols[1]:
        st.markdown("**Subcategory**")
    with header_cols[2]:
        st.markdown("**Topic**")
    with header_cols[3]:
        st.markdown("**Tonality Rule**")
    with header_cols[4]:
        st.markdown("**Submitted By**")
    with header_cols[5]:
        st.markdown("**Action**")
    
    st.divider()

    for idx, item in enumerate(filtered_data):
        row_cols = st.columns([2, 2, 2, 2, 1.5, 1])
        
        with row_cols[0]:
            st.write(item.get('category', ''))
        with row_cols[1]:
            st.write(item.get('subcategory') if item.get('subcategory') else '—')
        with row_cols[2]:
            st.write(item.get('topic', ''))
        with row_cols[3]:
            st.write(item.get('tonality', ''))
        with row_cols[4]:
            st.write(item.get('submitted_by', '—'))
        with row_cols[5]:
            if st.button("✏️ Edit", key=f"edit_btn_{item.get('id', idx)}", use_container_width=True):
                edit_taxonomy_modal(item, categories, subcategories, ALLOWED_TONALITY_OPTIONS)
        
        st.divider()
