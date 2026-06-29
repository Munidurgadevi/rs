"""
Streamlit Demo Application for AI Candidate Discovery & Ranking System.
Deploys the existing ranking pipeline as a web interface.
"""

import streamlit as st
import pandas as pd
import time
import tempfile
import os
from io import BytesIO

# Import existing ranking modules
from fast_rank import load_candidates, compute_scores_batch

# Streamlit page configuration
st.set_page_config(
    page_title="AI Candidate Discovery & Ranking System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from uploaded .docx file."""
    try:
        import docx
        doc = docx.Document(BytesIO(file_content))
        text = []
        for para in doc.paragraphs:
            text.append(para.text)
        return '\n'.join(text)
    except ImportError:
        st.error("python-docx not installed. Please install it to process .docx files.")
        return ""

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to a temporary path and return the path."""
    filename = uploaded_file.name
    if filename.endswith('.jsonl.gz'):
        suffix = '.jsonl.gz'
    else:
        file_ext = filename.split('.')[-1].lower()
        suffix = f'.{file_ext}'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name

def run_ranking_pipeline(candidates_path: str) -> tuple:
    """
    Run the existing ranking pipeline.
    Returns (results, execution_time, total_candidates, qualified_candidates)
    """
    start_time = time.time()
    
    # Load candidates
    candidates = load_candidates(candidates_path)
    total_candidates = len(candidates)
    
    # Progress bar placeholder
    progress_bar = st.progress(0, text="Loading candidates...")
    status_text = st.empty()
    
    status_text.text(f"Loaded {total_candidates} candidates. Computing scores...")
    progress_bar.progress(25, text="Scoring candidates...")
    
    # Compute scores using existing pipeline
    results = compute_scores_batch(candidates)
    qualified_candidates = len(results)
    
    progress_bar.progress(75, text="Sorting and ranking...")
    
    # Sort and rank
    results.sort(key=lambda x: (-x['score'], x['candidate_id']))
    
    # Assign ranks with strict decreasing scores
    final_results = []
    prev_score = 1.0
    
    for rank, item in enumerate(results[:100], 1):
        curr_score = min(prev_score, round(item['score'], 6))
        if curr_score >= prev_score:
            curr_score = round(prev_score - 0.00001, 6)
        prev_score = curr_score
        
        final_results.append({
            'candidate_id': item['candidate_id'],
            'rank': rank,
            'score': curr_score,
            'reasoning': item['reasoning']
        })
    
    progress_bar.progress(100, text="Ranking complete!")
    elapsed_time = time.time() - start_time
    
    return final_results, elapsed_time, total_candidates, qualified_candidates

def main():
    # Title and description
    st.title("🤖 AI Candidate Discovery & Ranking System")
    st.markdown("""
    <div style="margin-bottom: 20px;">
    This system ranks candidates for the Senior AI Engineer position using a sophisticated 
    multi-factor scoring algorithm. It evaluates semantic similarity, skill match, experience 
    relevance, career trajectory, behavioral signals, and availability.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for file uploads
    with st.sidebar:
        st.header("📁 Input Files")
        
        # Job description upload
        st.subheader("Job Description")
        job_file = st.file_uploader(
            "Upload Job Description (.docx or .txt)",
            type=['docx', 'txt'],
            key="job_desc"
        )
        
        # Candidates dataset upload
        st.subheader("Candidates Dataset")
        candidates_file = st.file_uploader(
            "Upload Candidates (.jsonl or .jsonl.gz)",
            type=['jsonl', 'gz'],
            key="candidates"
        )
        
        st.markdown("---")
        
        # Run button
        run_button = st.button(
            "🚀 Run Ranking",
            type="primary",
            use_container_width=True,
            disabled=(candidates_file is None)
        )
    
    # Main content area
    if candidates_file is not None:
        # Display job description if uploaded
        if job_file is not None:
            with st.expander("📄 Job Description Preview"):
                file_ext = job_file.name.split('.')[-1].lower()
                if file_ext == 'docx':
                    job_text = extract_text_from_docx(job_file.getbuffer())
                else:
                    job_text = job_file.getvalue().decode('utf-8')
                if job_text:
                    st.text_area("", job_text, height=200)
        
        # Run ranking when button is clicked
        if run_button:
            # Save uploaded candidates file
            candidates_path = save_uploaded_file(candidates_file)
            
            try:
                # Run the ranking pipeline
                with st.spinner("Running candidate ranking pipeline..."):
                    results, exec_time, total, qualified = run_ranking_pipeline(candidates_path)
                
                # Clear progress indicators
                st.empty()
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⏱️ Execution Time", f"{exec_time:.2f} seconds")
                with col2:
                    st.metric("📊 Total Processed", total)
                with col3:
                    st.metric("✅ Qualified Candidates", qualified)
                
                # Display success message
                st.success(f"Ranking completed! Top {len(results)} candidates ranked successfully.")
                
                # Display top 100 ranked candidates
                st.subheader("🏆 Top 100 Ranked Candidates")
                
                df = pd.DataFrame(results)
                df['score'] = df['score'].apply(lambda x: f"{x:.5f}")
                
                # Style the dataframe
                st.dataframe(
                    df,
                    use_container_width=True,
                    height=400,
                    column_config={
                        'candidate_id': 'Candidate ID',
                        'rank': st.column_config.NumberColumn('Rank', width=80),
                        'score': st.column_config.TextColumn('Score', width=120),
                        'reasoning': st.column_config.TextColumn('Reasoning', width='large')
                    }
                )
                
                # Download button
                csv_output = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_output,
                    file_name='team_id.csv',
                    mime='text/csv',
                    use_container_width=True
                )
                
            finally:
                # Clean up temporary file
                os.unlink(candidates_path)
    
    else:
        st.info("👈 Please upload a candidates dataset (.jsonl or .jsonl.gz) file to begin.")
        
        # Show expected format
        with st.expander("📋 Expected Data Format"):
            st.markdown("""
            **Candidates JSONL format:**
            ```json
            {"candidate_id": "CAND_000001", "profile": {...}, "skills": [...], "career_history": [...], "redrob_signals": {...}}
            ```
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "AI Candidate Discovery & Ranking System | Redrob Hackathon 2026"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()