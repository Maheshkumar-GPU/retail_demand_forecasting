import streamlit as st
import pickle
import numpy as np

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Big Mart Sales Predictor",
    page_icon="🛒",
    layout="wide"
)

# ----------------------------
# Load Model
# ----------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
h1 {
    color: #2c3e50;
}
.metric-card {
    background-color: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    text-align: center;
}
.footer {
    text-align: center;
    color: gray;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header Section
# ----------------------------
st.title("🛒 Big Mart Sales Prediction Dashboard")
st.caption("Industry-grade Machine Learning Application")
st.markdown("---")

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("🔧 Input Parameters")

product_id = st.sidebar.text_input("Product ID", help="Example: FDA15")

weight = st.sidebar.slider("Item Weight", 0.0, 50.0, 12.5)

fat_content = st.sidebar.selectbox(
    "Fat Content", ["Low Fat", "Regular"]
)

visibility = st.sidebar.slider(
    "Product Visibility", 0.0, 1.0, 0.05
)

product_type = st.sidebar.selectbox(
    "Product Type",
    [
        "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables",
        "Household", "Baking Goods", "Snack Foods", "Frozen Foods",
        "Breakfast", "Health and Hygiene", "Hard Drinks",
        "Canned", "Breads", "Starchy Foods", "Others", "Seafood"
    ]
)

mrp = st.sidebar.slider("MRP (₹)", 0, 300, 150)

outlet_id = st.sidebar.text_input("Outlet ID", help="Example: OUT049")

est_year = st.sidebar.slider("Establishment Year", 1980, 2025, 2000)

outlet_size = st.sidebar.selectbox("Outlet Size", ["Small", "Medium", "High"])
location_type = st.sidebar.selectbox("Location Tier", ["Tier 1", "Tier 2", "Tier 3"])
outlet_type = st.sidebar.selectbox(
    "Outlet Type",
    ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"]
)

# ----------------------------
# Encoding (same as before)
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

fat_content = fat_map[fat_content]
outlet_size = outlet_size_map[outlet_size]
location_type = location_map[location_type]
outlet_type = outlet_type_map[outlet_type]

product_type_encoded = hash(product_type) % 16
product_id_encoded = hash(product_id) % 100
outlet_id_encoded = hash(outlet_id) % 10

# ----------------------------
# Main Prediction Section
# ----------------------------
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🚀 Predict Sales", use_container_width=True):
        with st.spinner("Analyzing data & predicting sales..."):
            input_data = np.array([[
                product_id_encoded,
                weight,
                fat_content,
                visibility,
                product_type_encoded,
                mrp,
                outlet_id_encoded,
                est_year,
                outlet_size,
                location_type,
                outlet_type
            ]])

            prediction = model.predict(input_data)

        st.markdown(f"""
        <div class="metric-card">
            <h2>💰 Predicted Sales</h2>
            <h1>₹ {prediction[0]:,.2f}</h1>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown(
    "<div class='footer'>Built with ❤️ using Machine Learning & Streamlit</div>",
    unsafe_allow_html=True
)
