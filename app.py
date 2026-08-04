from datetime import date, datetime, timedelta
import json
import re
import time
from google import genai
from google.genai import types
import pandas as pd
from streamlit_supabase_auth import login_form
from supabase import Client, create_client
import streamlit as st

# ==========================================
# 1. PAGE CONFIG (Must be first Streamlit call)
# ==========================================
st.set_page_config(
    page_title="Master Taxonomy & AI Batch Classifier",
    page_icon="🌍",
    layout="wide",
)

# ==========================================
# 2. CUSTOM CSS STYLING & DATAFRAME FILTER THEME
# ==========================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #f8fafc 100%);
        font-family: 'Inter', sans-serif;
    }
    h1 {
        color: #0f172a;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.03), 0 4px 6px -2px rgba(0, 0, 0, 0.02);
        transition: all 0.3s ease;
    }
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
""",
    unsafe_allow_html=True,
)


# ==========================================
# 3. INITIALIZATIONS & CACHED RESOURCES
# ==========================================
@st.cache_resource
def get_gemini_client():
  return genai.Client(api_key=st.secrets["gemini"]["api_key"])


@st.cache_data
def load_taxonomy():
  try:
    xls = pd.ExcelFile("NEW SA Topic CAT Gemini.xlsx")
    return pd.read_excel(xls, xls.sheet_names[0])
  except FileNotFoundError:
    return None


@st.cache_resource
def init_connection() -> Client:
  url = st.secrets["supabase"]["url"]
  key = st.secrets["supabase"]["key"]
  return create_client(url, key)


supabase = init_connection()
taxonomy_df = load_taxonomy()

if taxonomy_df is None:
  st.warning(
      "⚠️ 'NEW SA Topic CAT Gemini.xlsx' was not found in the repository."
      " Please upload your taxonomy file below to proceed."
  )
  uploaded_tax_file = st.file_uploader(
      "Upload Master Taxonomy Excel", type=["xlsx"]
  )
  if uploaded_tax_file:
    taxonomy_df = pd.read_excel(uploaded_tax_file)
  else:
    st.stop()

ALLOWED_TONALITY_OPTIONS = [
    "All Tonalities",
    "Only Positive",
    "Only Negative",
    "Only Neutral",
    "Neutral & Negative",
]


def fetch_master_data():
  try:
    response = (
        supabase.table("taxonomy_entries")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )
    return response.data
  except Exception as e:
    st.error(f"Error fetching data from Supabase: {e}")
    return []


data = fetch_master_data()
categories = sorted(
    list(set([item.get("category") for item in data if item.get("category")]))
)
subcategories = sorted(
    list(
        set(
            [
                item.get("subcategory")
                for item in data
                if item.get("subcategory")
            ]
        )
    )
)
topics = sorted(
    list(set([item.get("topic") for item in data if item.get("topic")]))
)


# ==========================================
# 4. UTILITY & TEXT CLEANING FUNCTIONS
# ==========================================
def extract_main_topic(topic_str):
  """Strips leading NH numbers, Expressway names, or route designations from topics.

  Example: 'NH-24 Road damage' -> 'Road damage'
  """
  if not topic_str or pd.isna(topic_str):
    return ""
  text = str(topic_str).strip()
  cleaned = re.sub(
      r"^(?:NH-?[0-9A-Za-z]+|[A-Za-z]+(?:-[A-Za-z]+)*\s+(?:EW|Expressway|NH|HWY))\s*[-:]?\s*",
      "",
      text,
      flags=re.IGNORECASE,
  )
  if cleaned == text and "-" in text:
    parts = text.split("-", 1)
    if len(parts) > 1 and any(
        kw in parts[0].lower()
        for kw in ["nh", "expressway", "ew", "hwy", "delhi", "mumbai", "corridor"]
    ):
      cleaned = parts[1].strip()
  return cleaned.strip() if cleaned else text


# ==========================================
# 5. AI BATCH CLASSIFICATION CACHED FUNCTION
# ==========================================
@st.cache_data
def classify_batch_articles(articles_json_str, taxonomy_reference):
  client = get_gemini_client()
  articles_payload = json.loads(articles_json_str)

  prompt = f"""
    You are an expert media analyst and taxonomy classification engine.
    Analyze the batch of articles provided below and categorize each one strictly using ONLY the valid Categories, Subcategories, Topics, and Tonality rules in the Master Taxonomy Reference.
    
    Special Instructions:
    1. If the content includes the name of an expressway or highway, provide the National Highway number or National Expressway name. If only the highway name is mentioned without a number, deduce or search for the exact highway number.
    2. Identify the main location the article is primarily about along with its exact state.

    ### Master Taxonomy Reference:
    {taxonomy_reference}

    ### Articles Batch to Classify:
    {json.dumps(articles_payload)}

    Return your response strictly as a valid JSON array of objects with keys:
    "index", "Category", "Subcategory", "Topic", "Tonality", "Overall Tonality", "NH NO", "Location".
    """

  retries = 3
  for attempt in range(retries):
    try:
      response = client.models.generate_content(
          model="gemini-3.5-flash",
          contents=prompt,
          config=types.GenerateContentConfig(
              response_mime_type="application/json", temperature=0.1
          ),
      )
      return json.loads(response.text)
    except Exception as e:
      if attempt == retries - 1:
        error_results = []
        for item in articles_payload:
          error_results.append({
              "index": item["index"],
              "Category": "Error",
              "Subcategory": str(e),
              "Topic": "N/A",
              "Tonality": "N/A",
              "Overall Tonality": "N/A",
              "NH NO": "N/A",
              "Location": "N/A",
          })
        return error_results
      else:
        time.sleep(2**attempt)


# ==========================================
# 6. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🧭 Explore")
app_mode = st.sidebar.selectbox(
    "Choose Application Mode",
    ["📋 Master Taxonomy Manager", "🤖 MoRTH AI", "🔍 MoRTH QC"],
)


# ==========================================
# 7. MODAL DIALOGS
# ==========================================
@st.dialog("📝 Add New Taxonomy Entry")
def add_taxonomy_modal(categories, subcategories, ALLOWED_TONALITY_OPTIONS):
  with st.form("new_entry_form", clear_on_submit=True):
    op_name = st.text_input(
        "Your Name / User ID *", placeholder="Enter your full name..."
    )
    form_category = st.selectbox(
        "Category *",
        options=categories
        if categories
        else [
            "Interviews",
            "Grievances",
            "Projects and Infra",
            "Policies",
            "Analysis Report",
            "Rules Violation",
            "Technology",
            "Irregularities",
        ],
    )
    form_subcategory = st.selectbox(
        "Subcategory (Optional)", options=[""] + subcategories
    )
    form_topic = st.text_input("Topic *", placeholder="Enter topic name...")
    form_tonality = st.selectbox(
        "Tonality / Condition Rule *", options=ALLOWED_TONALITY_OPTIONS
    )

    submit_col, cancel_col = st.columns(2)
    with submit_col:
      submitted = st.form_submit_button(
          "🚀 Submit & Sync", type="primary", use_container_width=True
      )
    with cancel_col:
      cancelled = st.form_submit_button("❌ Cancel", use_container_width=True)

    if submitted:
      cleaned_topic = form_topic.strip()
      cleaned_name = op_name.strip()
      if not cleaned_name or not cleaned_topic:
        st.error("Please fill out your Name and the Topic field.")
      else:
        try:
          supabase.table("taxonomy_entries").insert({
              "category": form_category,
              "subcategory": form_subcategory
              if form_subcategory != ""
              else None,
              "topic": cleaned_topic,
              "tonality": form_tonality,
              "submitted_by": cleaned_name,
          }).execute()
          st.success("✨ Entry successfully added and synced!")
          st.rerun()
        except Exception as e:
          st.error(f"Failed to add entry: {e}")
    if cancelled:
      st.rerun()


@st.dialog("✏️ Edit Taxonomy Entry")
def edit_taxonomy_modal(
    item, categories, subcategories, ALLOWED_TONALITY_OPTIONS
):
  with st.form("edit_entry_form", clear_on_submit=False):
    editor_name = st.text_input(
        "Your Name / User ID (Editor) *", placeholder="Enter your full name..."
    )
    default_cat_idx = (
        categories.index(item.get("category"))
        if item.get("category") in categories
        else 0
    )
    edit_category = st.selectbox(
        "Category *", options=categories, index=default_cat_idx
    )

    sub_opts = [""] + subcategories
    default_sub_idx = (
        sub_opts.index(item.get("subcategory"))
        if item.get("subcategory") in sub_opts
        else 0
    )
    edit_subcategory = st.selectbox(
        "Subcategory (Optional)", options=sub_opts, index=default_sub_idx
    )

    edit_topic = st.text_input("Topic *", value=item.get("topic", ""))
    default_ton_idx = (
        ALLOWED_TONALITY_OPTIONS.index(item.get("tonality"))
        if item.get("tonality") in ALLOWED_TONALITY_OPTIONS
        else 0
    )
    edit_tonality = st.selectbox(
        "Tonality / Condition Rule *",
        options=ALLOWED_TONALITY_OPTIONS,
        index=default_ton_idx,
    )

    proceed_col, cancel_col = st.columns(2)
    with proceed_col:
      proceed_button = st.form_submit_button(
          "✅ Proceed & Update", type="primary", use_container_width=True
      )
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
              "subcategory": edit_subcategory
              if edit_subcategory != ""
              else None,
              "topic": cleaned_topic,
              "tonality": edit_tonality,
              "submitted_by": cleaned_name,
              "created_at": datetime.utcnow().isoformat(),
          }
          supabase.table("taxonomy_entries").update(update_payload).eq(
              "id", item.get("id")
          ).execute()
          st.success("Edited in the database successfully!")
          st.rerun()
        except Exception as e:
          st.error(f"Failed to update entry in database: {e}")
    if cancel_button:
      st.rerun()


# ==========================================
# 8. EXCEL-LIKE COLUMN FILTERING UTILITY (Matching user image layout)
# ==========================================
def render_excel_style_qc_section(df, display_columns, section_key_prefix):
  """Renders clean Excel-like multi-select filter dropdown boxes directly above each column,

  matching the provided interface layout exactly ("All Selected" dropdown boxes per column).
  """
  working_df = df.copy()
  available_cols = [c for c in display_columns if c in working_df.columns]
  working_df = working_df[available_cols]

  filtered_df = working_df.copy()

  # Render Excel-style column filter controls side-by-side matching headers
  st.markdown("### 🎛️ Column Filters")
  filter_cols = st.columns(len(available_cols))

  for idx, col_name in enumerate(available_cols):
    with filter_cols[idx]:
      unique_vals = sorted(
          [str(v) for v in working_df[col_name].dropna().unique()]
      )
      
      # Use multi-select mimicking Excel filter dropdown field
      selected_vals = st.multiselect(
          label=col_name,
          options=unique_vals,
          default=[],
          placeholder="All Selected",
          key=f"{section_key_prefix}_excel_col_{col_name}",
      )

      if selected_vals:
        filtered_df = filtered_df[
            filtered_df[col_name].astype(str).isin(selected_vals)
        ]

  st.divider()
  st.subheader(f"📊 Filtered Results ({len(filtered_df)} rows)")

  # Action Buttons Row
  btn_col1, btn_col2, _ = st.columns([1, 1, 3])

  with btn_col1:
    id_col_candidates = ["Article ID", "ArticleID", "article_id", "ID"]
    match_id_col = next(
        (c for c in id_col_candidates if c in filtered_df.columns), None
    )

    if match_id_col:
      unique_ids = (
          filtered_df[match_id_col].dropna().astype(str).unique().tolist()
      )
      ids_string = ", ".join(unique_ids)
      if st.button("📋 Copy Unique ID's", key=f"{section_key_prefix}_copy_btn"):
        st.code(ids_string, language="text")
        st.success(f"Copied {len(unique_ids)} unique IDs successfully!")

  with btn_col2:
    if not filtered_df.empty:
      csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
      st.download_button(
          label="📥 Download Dataset",
          data=csv_bytes,
          file_name=f"{section_key_prefix}_report.csv",
          mime="text/csv",
          key=f"{section_key_prefix}_download_btn",
      )

  st.dataframe(filtered_df, use_container_width=True)
  return filtered_df


# ==========================================
# 9. MAIN APP INTERFACE ROUTING
# ==========================================
if app_mode == "🤖 MoRTH AI":
  st.subheader("🤖 HELLO I AM MoRTH AI")
  st.markdown(
      "<p style='font-size: 15px; color: #475569;'>Choose whether to upload a"
      " spreadsheet or add content snippets sequentially below for instant"
      " AI classification.</p>",
      unsafe_allow_html=True,
  )
  st.divider()

  input_method = st.radio(
      "Select Input Source",
      ["📁 Upload Spreadsheet (CSV/Excel)", "✍️ Add Contents Sequentially"],
      horizontal=True,
  )

  df_input = None

  if input_method == "📁 Upload Spreadsheet (CSV/Excel)":
    uploaded_file = st.file_uploader(
        "Upload Excel or CSV containing your articles", type=["xlsx", "csv"]
    )
    if uploaded_file:
      df_input = (
          pd.read_csv(uploaded_file)
          if uploaded_file.name.endswith(".csv")
          else pd.read_excel(uploaded_file)
      )
      st.write("Preview of Uploaded Data:", df_input.head())

  else:
    st.markdown("### ✍️ Sequential Content Input")
    st.markdown(
        "<small style='color: #64748b;'>Paste your content snippet below and"
        " click the button to add it to your queue.</small>",
        unsafe_allow_html=True,
    )

    if "sequential_snippets" not in st.session_state:
      st.session_state.sequential_snippets = []
    if "cached_output_df" not in st.session_state:
      st.session_state.cached_output_df = None
    if "current_snippet_input" not in st.session_state:
      st.session_state.current_snippet_input = ""

    st.text_area(
        "Enter content snippet",
        key="current_snippet_input",
        placeholder="Paste your content snippet here...",
        height=100,
    )

    col_btn1, _ = st.columns([1, 4])
    with col_btn1:
      add_item_btn = st.button("➕ Add Content Item", use_container_width=True)

    if add_item_btn:
      snippet_text = st.session_state.current_snippet_input.strip()
      if snippet_text:
        st.session_state.sequential_snippets.append(snippet_text)
        st.session_state.cached_output_df = None
        st.session_state.current_snippet_input = ""
        st.rerun()
      else:
        st.warning("Please enter some content before adding.")

    if st.session_state.sequential_snippets:
      st.markdown("#### 📋 Added Contents Queue:")
      for idx, snip in enumerate(st.session_state.sequential_snippets):
        st.markdown(
            f"**{idx + 1}.** {snip[:120]}{'...' if len(snip) > 120 else ''}"
        )

      col_reset, _ = st.columns([1, 4])
      with col_reset:
        if st.button("🗑️ Clear All Items"):
          st.session_state.sequential_snippets = []
          st.session_state.cached_output_df = None
          st.rerun()

      df_input = pd.DataFrame(
          {"content_snippet": st.session_state.sequential_snippets}
      )

  if df_input is not None and not df_input.empty:
    batch_size = st.slider(
        "Batch Size (Articles per AI Request)", min_value=1, max_value=10, value=5
    )

    if st.button("🚀 Run AI Classification"):
      if "content_snippet" not in df_input.columns:
        st.error(
            "Error: Data must contain a column named 'content_snippet'."
        )
      else:

        def classify_in_batches(df_articles, taxonomy_df, batch_size):
          taxonomy_reference = taxonomy_df.to_string(index=False)
          results = []
          progress_bar = st.progress(0)
          total_rows = len(df_articles)

          for i in range(0, total_rows, batch_size):
            batch_df = df_articles.iloc[i : i + batch_size]
            articles_payload = []
            for idx, row in batch_df.iterrows():
              articles_payload.append({
                  "index": int(idx),
                  "content": str(row.get("content_snippet", "")),
              })

            articles_json_str = json.dumps(articles_payload)
            parsed_batch = classify_batch_articles(
                articles_json_str, taxonomy_reference
            )

            if isinstance(parsed_batch, list):
              results.extend(parsed_batch)
            progress_bar.progress(min((i + batch_size) / total_rows, 1.0))

          res_df = pd.DataFrame(results)
          if not res_df.empty and "index" in res_df.columns:
            res_df = res_df.sort_values(by="index").reset_index(drop=True)
            res_df = res_df.drop(columns=["index"])
          return pd.concat(
              [df_articles.reset_index(drop=True), res_df], axis=1
          )

        with st.spinner(
            "Processing batch chunks through Gemini 3.5 Flash..."
        ):
          st.session_state.cached_output_df = classify_in_batches(
              df_input, taxonomy_df, batch_size
          )

    if st.session_state.get("cached_output_df") is not None:
      st.success("✨ Classification complete!")
      st.dataframe(st.session_state.cached_output_df)

      csv_data = (
          st.session_state.cached_output_df.to_csv(index=False).encode("utf-8")
      )
      st.download_button(
          label="📥 Download Categorized Results as CSV",
          data=csv_data,
          file_name="categorized_articles_output.csv",
          mime="text/csv",
      )

elif app_mode == "🔍 MoRTH QC":
  st.subheader("🔍 MoRTH Quality Control (QC Dashboard)")
  st.markdown(
      "<p style='font-size: 15px; color: #475569;'>Upload your analysis Excel"
      " report below to run automated validations across specific QC"
      " sub-modules with embedded Excel-style header filters.</p>",
      unsafe_allow_html=True,
  )
  st.divider()

  qc_file = st.file_uploader(
      "Upload Master Analysis Report (Excel/CSV)", type=["xlsx", "csv"]
  )

  if qc_file:
    qc_df = (
        pd.read_csv(qc_file)
        if qc_file.name.endswith(".csv")
        else pd.read_excel(qc_file)
    )

    # Sub-tabs for MoRTH QC
    qc_tabs = st.tabs([
        "👤 Journalist QC",
        "🗂️ Topic & Taxonomy QC",
        "📸 Photo QC",
        "🗣️ Spokes QC",
        "⚖️ Conflicts",
        "⚠️ Blank Tonality QC",
    ])

    # 1. Journalist QC
    with qc_tabs[0]:
      st.markdown("### 👤 Journalist Quality Control")
      cols = ["Medium", "Article ID", "Analysis By", "Analysis By Bureau", "Journalist"]
      render_excel_style_qc_section(qc_df, cols, "journalist_qc")

    # 2. Topic Category and Sub Category QC
    with qc_tabs[1]:
      st.markdown("### 🗂️ Topic, Category & Sub-Category QC")
      st.markdown(
          "<small style='color: #64748b;'>Validating row attributes, leading"
          " expressway route prefixes, subcategory null handling, and database"
          " tonality rule matrices.</small>",
          unsafe_allow_html=True,
      )

      db_rules_map = {}
      for entry in data:
        db_top = extract_main_topic(entry.get("topic", ""))
        db_cat = str(entry.get("category", "")).strip().lower()
        db_sub = entry.get("subcategory")
        if db_sub is None or pd.isna(db_sub) or str(db_sub).strip().lower() in ["none", "nan", ""]:
          db_sub_norm = ""
        else:
          db_sub_norm = str(db_sub).strip().lower()
          
        db_ton_rule = str(entry.get("tonality", "All Tonalities")).strip()
        db_rules_map[(db_top.lower(), db_cat, db_sub_norm)] = db_ton_rule

      validation_rows = []
      for idx, row in qc_df.iterrows():
        raw_top = row.get("Topic", "")
        clean_top = extract_main_topic(raw_top).lower()
        
        r_cat = str(row.get("Category", "")).strip().lower()
        
        r_sub_raw = row.get("Sub Category1", "")
        if r_sub_raw is None or pd.isna(r_sub_raw) or str(r_sub_raw).strip().lower() in ["none", "nan", ""]:
          r_sub = ""
        else:
          r_sub = str(r_sub_raw).strip().lower()

        r_ton = str(row.get("Tonality", "")).strip()

        match_key = (clean_top, r_cat, r_sub)
        
        if match_key in db_rules_map:
          db_rule = db_rules_map[match_key]
          tonality_valid = True
          if db_rule == "Only Positive":
            if r_ton.lower() != "positive":
              tonality_valid = False
          elif db_rule == "Only Negative":
            if r_ton.lower() != "negative":
              tonality_valid = False
          elif db_rule == "Only Neutral":
            if r_ton.lower() != "neutral":
              tonality_valid = False
          elif db_rule == "Neutral & Negative":
            if r_ton.lower() not in ["neutral", "negative"]:
              tonality_valid = False

          if tonality_valid:
            rule_msg = "Correct Match"
          else:
            rule_msg = f"Tonality Rule Mismatch (Expected DB Rule: {db_rule})"
        else:
          matching_topics = [k[0] for k in db_rules_map.keys()]
          matching_cats = [k[1] for k in db_rules_map.keys()]
          
          if clean_top not in matching_topics:
            rule_msg = f"Mismatch: Topic '{extract_main_topic(raw_top)}' not found in DB"
          elif r_cat not in matching_cats:
            rule_msg = f"Mismatch: Category '{row.get('Category')}' does not match topic rules"
          else:
            rule_msg = "Mismatch: Subcategory or rule combination mismatch"

        validation_rows.append(rule_msg)

      val_df = qc_df.copy()
      val_df["Rule Logic"] = validation_rows

      cols_t = [
          "Medium",
          "Article ID",
          "Analysis By",
          "Topic",
          "Category",
          "Sub Category1",
          "Tonality",
          "Analysis By Bureau",
          "Rule Logic",
      ]
      render_excel_style_qc_section(val_df, cols_t, "topic_cat_qc")

    # 3. Photo QC
    with qc_tabs[2]:
      st.markdown("### 📸 Photo Quality Control")
      cols_photo = [
          "Article ID",
          "Medium",
          "Analysis By",
          "Analysis By Bureau",
          "Photo Mention",
          "Topic",
      ]
      render_excel_style_qc_section(qc_df, cols_photo, "photo_qc")

    # 4. Spokes QC
    with qc_tabs[3]:
      st.markdown("### 🗣️ Spokes Quality Control")
      spokes_df = qc_df.copy()
      flags = []
      for idx, row in spokes_df.iterrows():
        spoke_val = str(row.get("Spokes", "")).strip()
        quote_val = str(row.get("Quotes", "")).strip().lower()

        if spoke_val and spoke_val.lower() != "nan" and spoke_val != "":
          if quote_val in ["", "nan", "no", "blank", "none"]:
            flags.append("Missing Quotes")
          elif quote_val in ["yes", "true", "1"]:
            flags.append("Review Entry")
          else:
            flags.append("Missing Quotes")
        else:
          flags.append("OK")

      spokes_df["Flag"] = flags
      cols_spokes = [
          "Article ID",
          "Medium",
          "Analysis By",
          "Analysis By Bureau",
          "Spokes",
          "Quotes",
          "Flag",
      ]
      render_excel_style_qc_section(spokes_df, cols_spokes, "spokes_qc")

    # 5. Conflicts
    with qc_tabs[4]:
      st.markdown("### ⚖️ Tonality Conflicts QC")
      conflict_df = qc_df.copy()
      conflict_flags = []
      filtered_conflict_rows = []

      for idx, row in conflict_df.iterrows():
        overall_ton = str(row.get("Overall Tonality", "")).strip()
        row_ton = str(row.get("Tonality", "")).strip()

        is_missing = (
            not overall_ton
            or overall_ton.lower() in ["nan", "none", ""]
            or overall_ton.isspace()
        )
        is_mismatch = (
            not is_missing
            and row_ton
            and row_ton.lower() not in ["nan", "none", ""]
            and overall_ton.lower() != row_ton.lower()
        )

        if is_missing:
          conflict_flags.append("Missing Tonality")
          filtered_conflict_rows.append(row)
        elif is_mismatch:
          conflict_flags.append("Mismatched")
          filtered_conflict_rows.append(row)

      if filtered_conflict_rows:
        conf_result_df = pd.DataFrame(filtered_conflict_rows)
        conf_result_df["Flag"] = [
            (
                "Missing Tonality"
                if not str(o).strip() or str(o).lower() in ["nan", "none", ""]
                else "Mismatched"
            )
            for o in conf_result_df["Overall Tonality"]
        ]
        cols_conf = [
            "Article ID",
            "Medium",
            "Analysis By",
            "Analysis By Bureau",
            "Entity",
            "Category",
            "Sub Category1",
            "Overall Tonality",
            "Tonality",
            "Flag",
        ]
        render_excel_style_qc_section(conf_result_df, cols_conf, "conflicts_qc")
      else:
        st.success("✨ No tonality conflicts or missing values found!")

    # 6. Blank Tonality QC (Strictly Bureau is empty/blank and Topic is present - filtering out non-matching rows entirely)
    with qc_tabs[5]:
      st.markdown("### ⚠️ Blank Tonality / Bureau QC")
      st.markdown(
          "<small style='color: #64748b;'>Showing strictly rows matching the rule: Bureau is strictly blank/empty and Topic is present.</small>",
          unsafe_allow_html=True,
      )
      
      blank_df = qc_df.copy()
      filtered_blank = []

      for idx, row in blank_df.iterrows():
        topic_val = row.get("Topic", "")
        bureau_val = row.get("Analysis By Bureau", "")

        has_topic = topic_val is not None and not pd.isna(topic_val) and str(topic_val).strip() != "" and str(topic_val).strip().lower() not in ["nan", "none", ""]
        
        # Strict blank check: Only consider truly blank cells (ignoring text entries like 'None', 'NA', or 'nan')
        is_bureau_blank = bureau_val is None or pd.isna(bureau_val) or str(bureau_val).strip() == ""

        # Strict rule enforcement: Only keep rows where this rule is true
        if has_topic and is_bureau_blank:
          filtered_blank.append(row)

      if filtered_blank:
        blank_res_df = pd.DataFrame(filtered_blank)
        cols_blank = [
            "Article ID",
            "Medium",
            "Analysis By",
            "Overall Tonality",
            "Analysis By Bureau",
            "Topic",
        ]
        render_excel_style_qc_section(blank_res_df, cols_blank, "blank_qc")
      else:
        st.success(
            "✨ No rows match the Blank Tonality / Bureau rule (Bureau strictly blank & Topic present)."
        )

  else:
    st.info(
        "ℹ️ Please upload an analysis spreadsheet above to activate the QC"
        " modules."
    )

else:
  st.markdown("<h1>🌍 Master Taxonomy Manager</h1>", unsafe_allow_html=True)
  st.markdown(
      "<p style='font-size: 16px; color: #475569;'>Seamlessly search, filter,"
      " analyze, and manage shared classification rules.</p>",
      unsafe_allow_html=True,
  )
  st.divider()

  col_btn1, col_btn2, _ = st.columns([1, 1, 4])
  with col_btn1:
    if st.button("➕ Add New Entry", use_container_width=True):
      add_taxonomy_modal(categories, subcategories, ALLOWED_TONALITY_OPTIONS)
  with col_btn2:
    if data:
      df_export = pd.DataFrame(data)
      cols_to_drop = [
          col for col in ["id", "created_at"] if col in df_export.columns
      ]
      df_export = df_export.drop(columns=cols_to_drop)
      csv_data = df_export.to_csv(index=False).encode("utf-8")
      st.download_button(
          label="📥 Export Master CSV",
          data=csv_data,
          file_name="Master_Taxonomy_Report.csv",
          mime="text/csv",
          use_container_width=True,
      )

  st.markdown("### 🔍 Search Database Filters")
  with st.container():
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
      search_topic = st.selectbox("Topic Filter", options=["All"] + topics)
    with s_col2:
      search_category = st.selectbox(
          "Category Filter", options=["All"] + categories
      )
    with s_col3:
      search_subcategory = st.selectbox(
          "Subcategory Filter", options=["All"] + subcategories
      )

    f_col1, f_col2 = st.columns([2, 2])
    with f_col1:
      date_filter_option = st.selectbox(
          "📅 Filter by Creation/Update Date",
          options=[
              "All Time",
              "Today",
              "Two Weeks Old (Last 14 Days)",
              "One Month Old (Last 30 Days)",
              "Custom Range",
          ],
      )

    custom_date_range = None
    if date_filter_option == "Custom Range":
      with f_col2:
        custom_date_range = st.date_input(
            "Select Custom Date Range",
            value=(date.today() - timedelta(days=7), date.today()),
        )

  filtered_data = data
  if search_topic != "All":
    filtered_data = [
        item for item in filtered_data if item.get("topic") == search_topic
    ]
  if search_category != "All":
    filtered_data = [
        item for item in filtered_data if item.get("category") == search_category
    ]
  if search_subcategory != "All":
    filtered_data = [
        item
        for item in filtered_data
        if item.get("subcategory") == search_subcategory
    ]

  if date_filter_option != "All Time":
    now_utc = datetime.utcnow()
    filtered_by_date = []
    for item in filtered_data:
      created_at_str = item.get("created_at")
      if not created_at_str:
        continue
      try:
        created_dt = datetime.fromisoformat(
            created_at_str.replace("Z", "+00:00").split("+")[0]
        )
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
      except Exception:
        pass
    filtered_data = filtered_by_date

  st.divider()

  res_col1, res_col2 = st.columns([4, 1])
  with res_col1:
    st.subheader("📋 Filtered Taxonomy Records")
  with res_col2:
    st.markdown(
        f"<div style='text-align: right; background-color: #e0f2fe;"
        f" color: #0369a1; padding: 6px 14px; border-radius: 20px; font-weight:"
        f" bold; font-size: 13px;'>📊 {len(filtered_data)} entries found</div>",
        unsafe_allow_html=True,
    )

  if not filtered_data:
    st.info("ℹ️ No matching taxonomy records found matching your criteria.")
  else:
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
        st.write(item.get("category", ""))
      with row_cols[1]:
        st.write(
            item.get("subcategory") if item.get("subcategory") else "—"
        )
      with row_cols[2]:
        st.write(item.get("topic", ""))
      with row_cols[3]:
        st.write(item.get("tonality", ""))
      with row_cols[4]:
        st.write(item.get("submitted_by", "—"))
      with row_cols[5]:
        if st.button(
            "✏️ Edit",
            key=f"edit_btn_{item.get('id', idx)}",
            use_container_width=True,
        ):
          edit_taxonomy_modal(
              item, categories, subcategories, ALLOWED_TONALITY_OPTIONS
          )
      st.divider()
