import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="German Credit Predictor", layout="centered")
st.title("German Credit Risk Predictor////")
st.markdown("### Predict if a customer is **Good** or **Bad** credit risk")

# Load the model
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        model_info = pickle.load(f)
    return model_info["model"], model_info.get("classes")

model, classes = load_model()

# Sidebar for User Input
st.sidebar.header("Enter Customer Details")

def user_input_features():
    # 1. UI Widgets
    age = st.sidebar.slider("Age", 18, 75, 30)
    sex = st.sidebar.selectbox("Sex", ["male", "female"])
    job = st.sidebar.selectbox("Job Level", [0, 1, 2, 3], 
                               format_func=lambda x: ["Unemployed", "Unskilled", "Skilled", "Highly Skilled"][x])
    housing = st.sidebar.selectbox("Housing", ["own", "rent", "free"])
    
    # Mapping categorical text to the numbers to solve the 'string to float' error
    saving_map = {"little": 1, "moderate": 2, "quite rich": 3, "rich": 4}
    checking_map = {"little": 1, "moderate": 2, "rich": 3}
    housing_map = {"own": 1, "rent": 2, "free": 3}
    sex_map = {"male": 1, "female": 2}
    
    saving_account = st.sidebar.selectbox("Savings Account", list(saving_map.keys()))
    checking_account = st.sidebar.selectbox("Checking Account", list(checking_map.keys()))
    credit_amount = st.sidebar.number_input("Credit Amount (€)", min_value=100, max_value=20000, value=1000)
    duration = st.sidebar.slider("Duration (months)", 1, 72, 12)
    purpose = st.sidebar.selectbox("Purpose", 
        ['car', 'radio/TV', 'furniture/equipment', 'business', 'education', 
         'repairs', 'domestic appliances', 'vacation/others'])

    # 2. Final Dictionary - Using numbers to satisfy the model's math requirements
    data = {
        'Age (years)': age,
        'Sex & Marital Status': sex_map[sex],
        'Occupation': job,
        'Type of apartment': housing_map[housing],
        'Value Savings/Stocks': saving_map[saving_account],
        'Account Balance': checking_map[checking_account],
        'Credit Amount': credit_amount,
        'Duration of Credit (month)': duration,
        'Purpose': 1, # Numeric placeholder for the purpose column
        
        # Mandatory missing columns with numeric default values
        'Concurrent Credits': 1,
        'Guarantors': 1,
        'No of Credits at this Bank': 1,
        'Duration in Current address': 1,
        'Length of current employment': 1,
        'Telephone': 1,
        'Most valuable available asset': 1,
        'No of dependents': 1,
        'Payment Status of Previous Credit': 1,
        'Instalment percent': 1,
        'Creditability': 1 
    }
    
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Prediction Logic
if st.button("🔍 Predict Credit Risk", type="primary"):
    with st.spinner("Analyzing profile..."):
        try:
            # Predict
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]

            # Labeling logic
            if classes is not None:
                result = classes[prediction]
            else:
                result = "Good" if prediction == 1 else "Bad"

            # Display Results
            st.success(f"**Prediction: {str(result).upper()} Credit Risk**")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Good Probability", f"{probability[1]*100:.1f}%")
            with col2:
                st.metric("Bad Probability", f"{probability[0]*100:.1f}%")

            if str(result).lower() == "good":
                st.balloons()
            else:
                st.warning("⚠️ High Risk: Financial profile suggests potential risk.")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# Model Info Section
try:
    model_type = type(model.named_steps['model']).__name__
    st.info(f"Model used: **{model_type}**")
except:
    st.info("Model loaded successfully.")