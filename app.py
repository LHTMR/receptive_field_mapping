import streamlit as st

st.title("🧠 Receptive Field Mapping App")

st.markdown("""
Welcome to the **Receptive Field Mapping App** — a streamlined platform for analyzing behaviorally-relevant neural activity through high-precision video tracking and spike data alignment.

This tool was developed for neuroscience experiments involving sensory mapping.
""")

st.markdown("---")

st.header("🔍 What this app does")

st.markdown("""
- **Video Analysis with DeepLabCut**  
  Track motion of painted markers and colored filaments using a pre-trained pose estimation model.
  
- **Model Re-training**
  If your setup differs from the standard, you can re-train the model with your own data to improve accuracy.

- **Data Cleaning & Quality Control**  
  Check videos for clarity, marker visibility, and lighting consistency before processing.
  Outlier imputation is performed via machine-learning to smooth out any faults.

- **Feature Extraction & Bending Detection**  
  Automatically calculate bend angles from tracked filament coordinates over time.

- **Spike-Time Alignment**  
  Synchronize neural recordings with behavioral events, including touch timestamps and filament bending.

- **Interactive Visualization**  
  Plot synchronized motion and spike activity to explore patterns and correlations.
""")

st.markdown("---")

st.header("⚙️ Quick Background Summary")

st.markdown("""
This app uses **DeepLabCut** for marker tracking, aligns **neural spikes** to tracked behavioral events, and enables visualization through a simple **Streamlit interface** and **Plotly Express**.
""")

st.markdown("---")

st.header("✅ Before You Start")

st.markdown("""
Make sure your recordings follow the setup instructions in the **"Video Instructions"** page to ensure accurate analysis:
""")
