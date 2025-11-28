"""
🔒 Universal Redaction Tool - Streamlit Web Interface
Automated PII detection and redaction across multiple document formats
"""

import streamlit as st
import pandas as pd
from io import StringIO, BytesIO
import re

# Import redaction engine
import logic
from logic import redact_text
from accuracy import calculate_accuracy

# --------------------------
# Universal file readers
# --------------------------
def read_pdf(file):
    """Extract text from PDF files"""
    from pypdf import PdfReader
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return pd.DataFrame({"text": [text]})
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")
        return None

def read_docx(file):
    """Extract text from DOCX files"""
    from docx import Document
    try:
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return pd.DataFrame({"text": [text]})
    except Exception as e:
        st.error(f"❌ Error reading Word Doc: {e}")
        return None

def load_data(uploaded_file):
    """Universal loader: CSV, Excel, JSON, TXT, PDF, DOCX"""
    ext = uploaded_file.name.split('.')[-1].lower()

    if ext == 'csv':
        return pd.read_csv(uploaded_file)
    elif ext in ['xlsx', 'xls']:
        return pd.read_excel(uploaded_file)
    elif ext == 'json':
        try:
            return pd.read_json(uploaded_file)
        except ValueError:
            uploaded_file.seek(0)
            return pd.read_json(uploaded_file, lines=True)
    elif ext == 'txt':
        string_data = StringIO(uploaded_file.getvalue().decode("utf-8"))
        return pd.DataFrame({"text": [string_data.read()]})
    elif ext == 'pdf':
        return read_pdf(uploaded_file)
    elif ext == 'docx':
        return read_docx(uploaded_file)
    else:
        st.error(f"❌ Unsupported file format: .{ext}")
        return None

# --------------------------
# Visual redaction highlights
# --------------------------
def highlight_redaction(redacted_str, mode):
    """Highlight [ENTITY] or <ENTITY> tokens"""
    if not redacted_str:
        return ""
    
    # Highlight square brackets [NAME]
    highlighted = re.sub(
        r"(\[[A-Z0-9_]+\])",
        r"<span style='background-color:#ffe6f0; color:#b30059; font-weight:bold; padding:2px; border-radius:3px;'>\1</span>",
        redacted_str
    )
    
    # Highlight angular brackets <NAME> (for Redact mode highlights)
    highlighted = re.sub(
        r"(<[A-Z0-9_]+>)",
        r"<span style='background-color:#ffe6f0; color:#b30059; font-weight:bold; padding:2px; border-radius:3px;'>\1</span>",
        highlighted
    )
    
    return f"<pre style='white-space:pre-wrap; font-family:inherit'>{highlighted}</pre>"

# --------------------------
# Streamlit page config
# --------------------------
st.set_page_config(
    page_title="Universal Redaction Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.1em;
}
pre { background: #fafafa; padding: 10px; border-radius: 6px; }
.fixed-box { height: 200px; overflow-y: auto; padding: 8px; background: #fff; border: 1px solid #eee; border-radius:6px; }
</style>
""", unsafe_allow_html=True)

# --------------------------
# Sidebar: info only
# --------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    Automatically detects and redacts:
    - Names & Persons
    - Email Addresses
    - Phone Numbers
    - Credit Cards, IPs, URLs
    - Dates separately
    - And more
    """)
    st.markdown("---")
    st.header("📁 Supported Formats")
    st.markdown("""
    - **Structured:** CSV, Excel, JSON
    - **Documents:** PDF, DOCX, TXT
    """)

# --------------------------
# Main tabs
# --------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Live Demo", "📂 Batch Processing", "📊 Accuracy Check", "🔎 Entity Table"])

# ==========================
# TAB 1: LIVE REDACTION
# ==========================
with tab1:
    st.header("Live PII Redaction Demo")
    st.markdown("Enter text below and watch it get redacted in real-time")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Input Text")
        txt = st.text_area(
            "Paste your text here:",
            height=350,
            value="My name is John Doe, my email is john.doe@example.com, my phone is (555) 123-4567, my ID is ID-99882 and I live in New York City, ZIP 10001. small name example: my name is alex johnson.",
            label_visibility="collapsed"
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔍 Redact Live", type="primary", use_container_width=True):
                if txt.strip():
                    # Generate all 3 versions:
                    # 1. Redact (Empty strings + Smart Spacing)
                    redacted_text, _, analysis, _ = redact_text(txt, mode="redact")
                    # 2. Mask (Square brackets [TAG])
                    mask_text, _, _, _ = redact_text(txt, mode="mask")
                    # 3. Angular (Angular brackets <TAG> for highlights in redact mode)
                    angular_text, _, _, _ = redact_text(txt, mode="tag_angular")
                    
                    st.session_state['result_redacted'] = redacted_text
                    st.session_state['result_mask'] = mask_text
                    st.session_state['result_angular'] = angular_text
                    st.session_state['original'] = txt
                    st.session_state['analysis'] = analysis
                    st.session_state['analysis_count'] = len(analysis)
                else:
                    st.warning("⚠️ Please enter some text first!")

        with col_btn2:
            if st.button("🗑️ Clear", use_container_width=True):
                for k in ['result_redacted', 'result_mask', 'result_angular', 'original', 'analysis', 'analysis_count']:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()

    with col2:
        st.subheader("🔐 Output Controls")
        
        view_mode = st.radio("Output Mode:", options=["Redact (remove entities)", "Mask (show [ENTITY])"], index=0, horizontal=False)
        mode_value_tab1 = "redact" if view_mode.startswith("Redact") else "mask"

        st.markdown("### 🔐 Redacted Output")
        if 'result_redacted' in st.session_state:
            
            # --- OUTPUT SWAP LOGIC ---
            if mode_value_tab1 == "redact":
                # Redact Mode:
                # Output Box: Clean text (e.g. "My name is .")
                # Visual Box: Angular Tags (e.g. "My name is <PERSON>.")
                out_text = st.session_state['result_redacted']
                vis_text = st.session_state['result_angular']
            else:
                # Mask Mode:
                # Output Box: Square Tags (e.g. "My name is [PERSON].")
                # Visual Box: Square Tags (e.g. "My name is [PERSON].")
                out_text = st.session_state['result_mask']
                vis_text = st.session_state['result_mask']
            
            st.text_area("Safe Text:", value=out_text, height=200, label_visibility="collapsed")

            st.markdown("### 🎨 Visual Highlight")
            st.markdown(highlight_redaction(vis_text, mode_value_tab1), unsafe_allow_html=True)

            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("Items Detected", st.session_state.get('analysis_count', 0))
            with col_stat2:
                try:
                    reduction_pct = round((1 - len(st.session_state['result_redacted']) / max(1, len(st.session_state['original']))) * 100, 1)
                except Exception:
                    reduction_pct = 0.0
                st.metric("Text Reduction", f"{reduction_pct}%")

            format_choice = st.selectbox("Download format:", options=["txt", "csv", "xlsx"], index=0)
            if st.button("💾 Download Output", use_container_width=True):
                if format_choice == "txt":
                    st.download_button(
                        label="Download TXT",
                        data=out_text,
                        file_name="redaction_output.txt",
                        mime="text/plain"
                    )
                elif format_choice == "csv":
                    csv_bytes = ("text\n" + out_text.replace("\n", "\\n") + "\n").encode("utf-8")
                    st.download_button(
                        label="Download CSV",
                        data=csv_bytes,
                        file_name="redaction_output.csv",
                        mime="text/csv"
                    )
                elif format_choice == "xlsx":
                    out_df = pd.DataFrame({"text": [out_text]})
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine="openpyxl") as writer:
                        out_df.to_excel(writer, index=False)
                    st.download_button(
                        label="Download XLSX",
                        data=output.getvalue(),
                        file_name="redaction_output.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        else:
            st.info("👆 Enter text and click 'Redact Live' to see results")

# ==========================
# TAB 2: BATCH PROCESSING
# ==========================
with tab2:
    st.header("Batch File Processing")
    st.markdown("Upload a file and redact all PII in one go")

    col_info, col_support = st.columns(2)
    with col_info:
        st.info("Supports: **CSV, XLSX, JSON, PDF, DOCX, TXT**")

    if "batch_df" not in st.session_state:
        st.session_state.batch_df = None
    if "batch_filename" not in st.session_state:
        st.session_state.batch_filename = ""

    uploaded_file = st.file_uploader(
        "Upload your file:",
        type=['csv', 'xlsx', 'xls', 'json', 'pdf', 'docx', 'txt']
    )

    if uploaded_file:
        if uploaded_file.name != st.session_state.batch_filename:
            st.session_state.batch_df = None
            st.session_state.batch_filename = uploaded_file.name
            
        st.success(f"✅ File uploaded: **{uploaded_file.name}**")
        df_loaded = load_data(uploaded_file)
        
        if df_loaded is not None:
            st.write(f"📊 Loaded {len(df_loaded)} rows × {len(df_loaded.columns)} columns")
            
            with st.expander("👀 Preview Data", expanded=False):
                st.dataframe(df_loaded.head(3), use_container_width=True)

            options = df_loaded.columns.tolist()
            default_index = options.index("text") if "text" in options else 0
            
            target_col = st.multiselect(
                "Select column(s) to redact (pick one or more):",
                options,
                default=[options[default_index]]
            )

            if st.button("⚡ Process Entire File", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_redacted = []
                results_mask = []

                for i, row in df_loaded.iterrows():
                    combined = []
                    for c in target_col:
                        if pd.notna(row[c]):
                            combined.append(str(row[c]))
                    original_text = " ".join(combined)
                    
                    r_out, _, _, _ = redact_text(original_text, mode="redact")
                    m_out, _, _, _ = redact_text(original_text, mode="mask")
                    
                    results_redacted.append(r_out)
                    results_mask.append(m_out)
                    
                    progress = (i + 1) / len(df_loaded)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {i + 1}/{len(df_loaded)} rows...")

                status_text.text("✅ Processing complete!")
                
                df_final = df_loaded.copy()
                df_final['original_text'] = df_final[target_col].astype(str).agg(' '.join, axis=1) if len(target_col) > 1 else df_final[target_col[0]].astype(str)
                df_final['redacted_text'] = results_redacted
                df_final['mask_text'] = results_mask
                
                st.session_state.batch_df = df_final

            if st.session_state.batch_df is not None:
                st.markdown("### 📋 Results Preview")
                
                st.dataframe(
                    st.session_state.batch_df[['original_text', 'redacted_text', 'mask_text']].head(5),
                    use_container_width=True,
                    column_config={
                        "original_text": st.column_config.TextColumn("Original", width="medium"),
                        "redacted_text": st.column_config.TextColumn("Redacted (Clean)", width="medium"),
                        "mask_text": st.column_config.TextColumn("Masked (Tags)", width="medium")
                    }
                )

                col_down_left, col_down_right = st.columns([2,3])
                with col_down_left:
                    file_format = st.selectbox("Choose download format:", options=["csv", "xlsx", "txt", "pdf"])
                
                with col_down_right:
                    data_to_download = None
                    file_name = ""
                    mime_type = ""
                    
                    if file_format == "csv":
                        data_to_download = st.session_state.batch_df.to_csv(index=False).encode('utf-8')
                        file_name = f"redacted_{st.session_state.batch_filename.split('.')[0]}.csv"
                        mime_type = "text/csv"
                        
                    elif file_format == "xlsx":
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine="openpyxl") as writer:
                            st.session_state.batch_df.to_excel(writer, index=False)
                        data_to_download = output.getvalue()
                        file_name = f"redacted_{st.session_state.batch_filename.split('.')[0]}.xlsx"
                        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        
                    elif file_format == "txt":
                        txt_lines = []
                        for _, r in st.session_state.batch_df.iterrows():
                            txt_lines.append(f"ORIGINAL:\n{r['original_text']}\nREDACTED:\n{r['redacted_text']}\nMASK:\n{r['mask_text']}\n\n---\n")
                        data_to_download = "\n".join(txt_lines)
                        file_name = f"redacted_{st.session_state.batch_filename.split('.')[0]}.txt"
                        mime_type = "text/plain"
                        
                    elif file_format == "pdf":
                        try:
                            from reportlab.lib.pagesizes import letter
                            from reportlab.pdfgen import canvas
                            output = BytesIO()
                            c = canvas.Canvas(output, pagesize=letter)
                            width, height = letter
                            y = height - 40
                            for _, r in st.session_state.batch_df.iterrows():
                                lines = [
                                    "ORIGINAL: " + str(r['original_text']),
                                    "REDACTED: " + str(r['redacted_text']),
                                    "MASK: " + str(r['mask_text']),
                                    "-" * 60
                                ]
                                for line in lines:
                                    c.drawString(40, y, line[:100].replace('\n', ' '))
                                    y -= 12
                                    if y < 60:
                                        c.showPage()
                                        y = height - 40
                            c.save()
                            data_to_download = output.getvalue()
                            file_name = f"redacted_{st.session_state.batch_filename.split('.')[0]}.pdf"
                            mime_type = "application/pdf"
                        except ImportError:
                            st.error("ReportLab not installed")

                    if data_to_download:
                        st.download_button(
                            label="💾 Download Processed File",
                            data=data_to_download,
                            file_name=file_name,
                            mime=mime_type,
                            use_container_width=True
                        )

# ==========================
# TAB 3: ACCURACY EVALUATION
# ==========================
with tab3:
    st.header("Accuracy Evaluation")
    st.markdown("Compare redaction output against ground truth")

    st.info("Use this to evaluate redaction quality and fine-tune the system")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📌 Original Text")
        original = st.text_area("Original text:", height=200)
    with col2:
        st.subheader("🔐 Redacted Text")
        redacted = st.text_area("Your redaction:", height=200)
    with col3:
        st.subheader("✅ Ground Truth")
        ground_truth = st.text_area("Expected redaction:", height=200)

    if st.button("📊 Calculate Accuracy", type="primary", use_container_width=True):
        if original and redacted and ground_truth:
            results = calculate_accuracy(original, redacted, ground_truth)
            st.markdown("### 📈 Results")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Match Accuracy", f"{results['Match Accuracy']}%")
            with col_m2:
                st.metric("Content Preservation", f"{results['Content Preservation']}%")
            avg_score = (results['Match Accuracy'] + results['Content Preservation']) / 2
            if avg_score >= 90:
                st.success("✅ Excellent redaction quality!")
            elif avg_score >= 75:
                st.info("✓ Good redaction quality")
            else:
                st.warning("⚠️ May need adjustment")
        else:
            st.warning("⚠️ Please fill in all three text areas")

# ==========================
# TAB 4: ENTITY TABLE (FIXED)
# ==========================
with tab4:
    st.header("Entity Visualization Table")
    st.markdown("All detected entities from the last Live/Batch run are shown here with exact spans.")

    analysis = None
    original_text = ""

    # Live Demo priority
    if 'analysis' in st.session_state and st.session_state['analysis']:
        analysis = st.session_state['analysis']
        original_text = st.session_state.get('original', "")
    # Batch fallback
    elif 'batch_df' in st.session_state and st.session_state.batch_df is not None:
        first_row = st.session_state.batch_df.iloc[0]
        original_text = first_row.get('original_text', "")
        if original_text:
            # FIX: Unpack 4 values, keeping only the analysis
            _, _, analysis, _ = redact_text(original_text, mode="redact")

    if analysis and original_text:
        rows = []
        for r in analysis:
            entity_text = original_text[r.start:r.end] if original_text else "<original not available>"
            rows.append({
                "Entity": r.entity_type,
                "Extracted Text": entity_text,
                "Start": r.start,
                "End": r.end,
                "Score": round(r.score, 2)
            })
        df_entities = pd.DataFrame(rows)
        st.dataframe(df_entities, use_container_width=True)
    else:
        st.info("Run a live redaction or batch process first to populate the entity table.")

# --------------------------
# Footer
# --------------------------
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>🔒 Universal Redaction Tool V1.0 <br></strong> &copy; 2025 | Developed by Super Syancs Team</p>
</div>
""", unsafe_allow_html=True)