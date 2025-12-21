import streamlit as st
import pickle
import numpy as np

# ----------------------------
# Load trained model
# ----------------------------
with open ('model.pkl','rb') as f:
  model = pickle.load(f)

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="Big Mart Sales Prediction",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 Big Mart Sales Prediction")
st.write("Predict item outlet sales using machine learning")

st.markdown("---")

# ----------------------------
# User Inputs
# ----------------------------
item_weight = st.number_input("Item Weight", min_value=0.0, max_value=50.0, step=0.1)

item_fat_content = st.selectbox(
    "Item Fat Content",
    ["Low Fat", "Regular"]
)

item_visibility = st.number_input(
    "Item Visibility",
    min_value=0.0,
    max_value=1.0,
    step=0.001
)

item_type = st.selectbox(
    "Item Type",
    [
        "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables",
        "Household", "Baking Goods", "Snack Foods", "Frozen Foods",
        "Breakfast", "Health and Hygiene", "Hard Drinks",
        "Canned", "Breads", "Starchy Foods", "Others", "Seafood"
    ]
)

item_mrp = st.number_input(
    "Item MRP",
    min_value=0.0,
    max_value=300.0,
    step=1.0
)

outlet_size = st.selectbox(
    "Outlet Size",
    ["Small", "Medium", "High"]
)

outlet_location_type = st.selectbox(
    "Outlet Location Type",
    ["Tier 1", "Tier 2", "Tier 3"]
)

outlet_type = st.selectbox(
    "Outlet Type",
    [
        "Grocery Store",
        "Supermarket Type1",
        "Supermarket Type2",
        "Supermarket Type3"
    ]
)

# ----------------------------
# Encoding (SIMPLE EXAMPLE)
# NOTE: Must match training encoding
# ----------------------------
fat_map = {"Low Fat": 0, "Regular": 1}
outlet_size_map = {"Small": 0, "Medium": 1, "High": 2}
location_map = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}
outlet_type_map = {
    "Grocery Store": 0,
    "Supermarket Type1": 1,
    "Supermarket Type2": 2,
    "Supermarket Type3": 3
}

item_fat_content = fat_map[item_fat_content]
outlet_size = outlet_size_map[outlet_size]
outlet_location_type = location_map[outlet_location_type]
outlet_type = outlet_type_map[outlet_type]

# ⚠ item_type encoding depends on your training
item_type_encoded = hash(item_type) % 16  # placeholder (better use same encoder)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Sales"):
    input_data = np.array([[
        item_weight,
        item_fat_content,
        item_visibility,
        item_type_encoded,
        item_mrp,
        outlet_size,
        outlet_location_type,
        outlet_type
    ]])

    prediction = model.predict(input_data)

    st.success(f"💰 Predicted Sales: ₹ {prediction[0]:.2f}")

st.markdown("---")
st.caption("Developed using Machine Learning & Streamlit")