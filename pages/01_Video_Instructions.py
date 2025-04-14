import streamlit as st
from PIL import Image
import os

# Set path to assets folder
ASSETS_PATH = "C:/Python Programming/LIU/receptive_field_mapping_app/assets"

# App title
st.title("🎥 Recording Instructions and Requirements")
st.markdown("For the recording to be properly recognized by the AI model, a few things need to be marked correctly to allow for accurate predicictions and follow-up post-processing.")

# --- Pre-filming requirements ---
st.header("📋 Pre-filming Requirements")
st.markdown("""
The model is designed to detect:
- 4 dots on the skin that represent the 4 corners of a square with sides of **1 or 2 cm**.
- A filament with 3 **separate color zones**, allowing it to distinguish **6 points for bending**.
""")

st.markdown("#### ✅ Lighting and Camera")
st.markdown("""
- Use **static, clean white lighting** for the subject.
- Ensure a **stationary camera** throughout the recording.
""")
st.markdown("#### ✅ Skin Dot Marking")
st.markdown("""
- Paint **4 dots** in the corners of a square using a **bright, opaque green** marker.
    - Recommended: [Posca paint pens](https://www.posca.com/en/product/pc-5m/)

- Example of bright green dots on skin, in a well-lit setting:
""")
dots_img = Image.open(os.path.join(ASSETS_PATH, "dots_example.png"))
st.image(dots_img, caption="Example: Bright green dots on skin", use_container_width=True)

st.markdown("#### ✅ Filament Marking")
st.markdown("""
- Paint the filament with **3 distinct opaque colors**.
- This allows the model to detect **6 separate points** for bend analysis.

- Example of a filament painted white and dark blue:
""")
filament_img = Image.open(os.path.join(ASSETS_PATH, "filament_example.png"))
st.image(filament_img, caption="Example: Colored filament with clear zones", use_container_width=True)

st.markdown("#### ✅ Clean the Skin")
st.markdown("- Ensure no old marks or blemishes interfere with detection.")

# --- During-filming requirements ---
st.header("🎬 During-Filming Requirements")

st.markdown("""
- Start the video with **5 touches** at the hotspot — one per second — for synchronization with neuron data.
- Ensure **no other filaments are visible** in the frame.
- Confirm that the **filament is clearly visible and not blurry** during bending.
""")

st.markdown("#### 📸 Examples")
col1, col2, col3 = st.columns(3)
with col1:
    bad1 = Image.open(os.path.join(ASSETS_PATH, "bad_bend_example_1.png"))
    st.image(bad1, caption="❌ Blurry region of interest")
with col2:
    bad2 = Image.open(os.path.join(ASSETS_PATH, "bad_bend_example_2.png"))
    st.image(bad2, caption="❌ Poor bend angle")
with col3:
    good = Image.open(os.path.join(ASSETS_PATH, "good_bend_example.png"))
    st.image(good, caption="✅ clear, visible bend")

st.success("Double-check recordings to ensure clear visibility of bend and proper lighting.")
