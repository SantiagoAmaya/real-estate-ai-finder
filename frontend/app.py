"""
Streamlit Frontend for Real Estate AI Finder
Modified: API Only Version (No Cache, No Local)
"""
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import time

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Real Estate AI Finder",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS
# ============================================================================

# Backend URL (change in production)
BACKEND_URL = st.secrets.get("BACKEND_URL", "http://localhost:8000")

# ============================================================================
# SESSION STATE
# ============================================================================

if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

if 'search_results' not in st.session_state:
    st.session_state.search_results = None

if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def search_properties(
    query: str,
    api_key: str,
    backend: str,
    vision_mode: str,
    use_vision_agent: bool,
    skip_scrape: bool,
    max_results: int
):
    """Call backend API to search properties"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/search",
            json={
                "query": query,
                "api_key": api_key,
                "backend": backend,
                "vision_mode": vision_mode if vision_mode != "None" else None,
                "use_vision_agent": use_vision_agent,
                "skip_scrape": skip_scrape,
                "max_results": max_results
            },
            timeout=600  # 5 minutes timeout
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            return None, f"Error {response.status_code}: {error_detail}"
            
    except requests.exceptions.Timeout:
        return None, "Request timed out (>5 minutes). Try with fewer properties."
    except requests.exceptions.ConnectionError:
        return None, f"Cannot connect to backend at {BACKEND_URL}. Is it running?"
    except Exception as e:
        return None, f"Error: {str(e)}"


def format_price(price):
    """Format price with thousand separators"""
    if price is None:
        return "N/A"
    return f"€{price:,}"


def format_cost(cost_eur):
    """Format cost in EUR"""
    return f"€{cost_eur:.3f}"


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.title("🏠 Real Estate AI Finder")
    st.markdown("*Find properties that match complex requirements using AI*")
    
    # ========================================================================
    # SIDEBAR: Configuration
    # ========================================================================
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key
        st.markdown("### 🔑 Anthropic API Key")
        api_key_input = st.text_input(
            "Enter your Anthropic API key",
            type="password",
            value=st.session_state.api_key,
            help="Get your API key from: https://console.anthropic.com/"
        )
        
        if api_key_input:
            st.session_state.api_key = api_key_input
            st.success("✅ API key set")
        else:
            st.warning("⚠️ API key required")
        
        st.markdown("---")
        
        # Settings Simplified
        with st.expander("🔧 Settings", expanded=True):
            
            # Backend hardcoded to API (Local removed)
            backend = "api" 
            
            # Vision Mode filtered (Removed claude_primary and qwen_only)
            vision_mode = st.selectbox(
                "Vision Analysis Mode",
                options=["claude_only", "None"],
                index=0,
                help="claude_only: Full AI analysis using Claude Vision. None: Text-only analysis (cheaper)."
            )
            
            use_vision_agent = st.checkbox(
                "Use Vision Decision Agent",
                value=True,
                help="LLM decides intelligently when to analyze images to save costs"
            )
            
            # REMOVED: skip_scrape checkbox
            # We will force skip_scrape=False in the call below
            
            ax_results = st.slider(
                "Max Results to Show",
                min_value=3,   # Bajar mínimo
                max_value=15,  # Bajar máximo
                value=5,       # Poner por defecto 5 (Mucho más rápido)
                help="Number of top properties to display"
            )
        
        # Cost estimate logic updated
        st.markdown("---")
        st.markdown("### 💰 Cost Estimate")
        
        if vision_mode == "None":
            cost_estimate = 0.003  # Text only cost approx
        else:
            # Claude Only cost approx
            cost_estimate = 0.078
        
        st.metric(
            "Per Property",
            format_cost(cost_estimate),
            help="Approximate cost per property analyzed"
        )
        
        # Help updated
        st.markdown("---")
        st.markdown("### ℹ️ Help")
        st.markdown("""
        **Example queries:**
        - Local entrada independiente Barcelona
        - Piso 3 habitaciones luminoso máx 300k
        
        **Mode:**
        - **Real-Time Search**: Always searches for fresh data.
        
        **Vision Modes:**
        - **claude_only**: Best quality (Analyzing images with Claude)
        - **None**: Text-only analysis (Faster/Cheaper)
        """)
    
    # ========================================================================
    # MAIN AREA: Search
    # ========================================================================
    
    # Check if API key is set
    if not st.session_state.api_key:
        st.info("👈 Please enter your Anthropic API key in the sidebar to start")
        st.stop()
    
    # Search form
    st.markdown("## 🔍 Search Properties")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "What are you looking for?",
            placeholder="e.g., Local con entrada independiente Barcelona máximo 250mil",
            label_visibility="collapsed"
        )
    
    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Search execution
    if search_button:
        if not query:
            st.error("Please enter a search query")
        else:
            with st.spinner("🔄 Searching and analyzing properties..."):
                # Progress indicators
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                progress_text.text("📝 Parsing your query...")
                progress_bar.progress(0.2)
                time.sleep(0.5)
                
                progress_text.text("🔍 Scraping fresh properties...")
                progress_bar.progress(0.4)
                
                # Call backend
                result, error = search_properties(
                    query=query,
                    api_key=st.session_state.api_key,
                    backend=backend, # Will always be "api"
                    vision_mode=vision_mode,
                    use_vision_agent=use_vision_agent,
                    skip_scrape=False, # FORCED TO FALSE: Always scrape fresh
                    max_results=max_results
                )
                
                progress_text.text("🤖 Analyzing with AI...")
                progress_bar.progress(0.8)
                
                if error:
                    st.error(f"❌ {error}")
                    progress_text.empty()
                    progress_bar.empty()
                else:
                    progress_bar.progress(1.0)
                    progress_text.text("✅ Complete!")
                    time.sleep(0.5)
                    progress_text.empty()
                    progress_bar.empty()
                    
                    # Store results
                    st.session_state.search_results = result
                    
                    # Add to history
                    st.session_state.search_history.append({
                        'query': query,
                        'timestamp': datetime.now(),
                        'num_results': len(result['properties']),
                        'cost': result['cost_summary'].get('total_cost_eur', 0)
                    })
                    
                    st.success(f"✅ Found {result['total_found']} properties in {result['processing_time_seconds']:.1f}s")
    
    # ========================================================================
    # RESULTS DISPLAY
    # ========================================================================
    
    if st.session_state.search_results:
        results = st.session_state.search_results
        
        st.markdown("---")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Properties Found",
                results['total_found']
            )
        
        with col2:
            st.metric(
                "Processing Time",
                f"{results['processing_time_seconds']:.1f}s"
            )
        
        with col3:
            cost = results['cost_summary'].get('total_cost_eur', 0)
            st.metric(
                "Total Cost",
                format_cost(cost)
            )
        
        with col4:
            vision_analyzed = sum(1 for p in results['properties'] if p['had_vision_analysis'])
            st.metric(
                "Vision Analyzed",
                f"{vision_analyzed}/{len(results['properties'])}"
            )
        
        st.markdown("---")
        
        # Properties table
        st.markdown("## 📊 Top Results")
        
        # Display each property
        for i, prop in enumerate(results['properties'], 1):
            with st.container():
                # Property header
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"### {i}. Property #{prop['id']}")
                    st.markdown(f"📍 {prop['location']}")
                
                with col2:
                    st.metric("Match Score", f"{prop['score']:.3f}")
                
                with col3:
                    st.metric("Price", format_price(prop['price']))
                
                # Property details
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Description
                    st.markdown("**Description:**")
                    st.write(prop['description'])
                    
                    # Matched features
                    if prop['matched_features']:
                        st.markdown("**✅ Matched Features:**")
                        st.markdown(", ".join([f"`{f}`" for f in prop['matched_features']]))
                    
                    # Missing features
                    if prop['missing_features']:
                        st.markdown("**❌ Missing Features:**")
                        st.markdown(", ".join([f"`{f}`" for f in prop['missing_features']]))
                
                with col2:
                    # Property specs
                    specs = []
                    if prop['size_m2']:
                        specs.append(f"📏 {prop['size_m2']} m²")
                    if prop['rooms']:
                        specs.append(f"🛏️ {prop['rooms']} rooms")
                    if prop['had_vision_analysis']:
                        specs.append("👁️ Vision analyzed")
                    
                    if specs:
                        st.markdown("**Specs:**")
                        for spec in specs:
                            st.markdown(f"- {spec}")
                    
                    # Images
                    if prop['images']:
                        st.markdown("**Images:**")
                        for img_url in prop['images'][:2]:  # Show first 2
                            try:
                                st.image(img_url, width=200)
                            except:
                                pass
                    
                    # Link to property
                    if prop.get('url'):
                        st.markdown(f"[🔗 View on Fotocasa]({prop['url']})")
                
                st.markdown("---")
    
    # ========================================================================
    # SEARCH HISTORY
    # ========================================================================
    
    if st.session_state.search_history:
        with st.expander("📜 Search History", expanded=False):
            history_df = pd.DataFrame([
                {
                    'Query': h['query'][:50] + "..." if len(h['query']) > 50 else h['query'],
                    'Results': h['num_results'],
                    'Cost': format_cost(h['cost']),
                    'Time': h['timestamp'].strftime("%H:%M:%S")
                }
                for h in reversed(st.session_state.search_history[-10:])  # Last 10
            ])
            st.dataframe(history_df, use_container_width=True, hide_index=True)


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()