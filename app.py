# Importing
import pickle
import streamlit as st

# Load models
model_log_reg = pickle.load(open('/content/Breast_Cancer_Model_Log_Reg_98_percent.pkl', 'rb'))
model_knn = pickle.load(open('/content/Breast_Cancer_Model_KNN_96_percent.pkl', 'rb'))

# Mean and Standard Deviation dictionaries for scaling
mean = {'radius_mean': 14.127291739894552, 
        'perimeter_mean': 91.96903339191564, 
        'area_mean': 654.8891036906855, 
        'concave points_mean': 0.04891914586994726, 
        'radius_worst': 16.269189806678387, 
        'perimeter_worst': 107.26121265377857, 
        'area_worst': 880.5831282952548, 
        'concave points_worst': 0.11460622319859401, 
        'diagnosis': 0.37258347978910367
}

std_dev = {'radius_mean': 3.524048826212078, 
           'perimeter_mean': 24.2989810387549, 
           'area_mean': 351.9141291816527, 
           'concave points_mean': 0.03880284485915359, 
           'radius_worst': 4.833241580469324, 
           'perimeter_worst': 33.60254226903635, 
           'area_worst': 569.3569926699492, 
           'concave points_worst': 0.0657323411959421, 
           'diagnosis': 0.4839179564031686
}

# Project description on web interface
st.title("Breast Cancer Type Prediction")
st.header("Leveraging Machine Learning for Early Detection")
st.markdown("""
    This application uses advanced machine learning algorithms to predict breast cancer types (Benign or Malignant)
    based on various features. After testing multiple models, we have selected the most reliable
    model to ensure accurate predictions with hyperparameter tuning and cross-validation for improved reliability.
    Our goal is to assist in early diagnosis and improve patient outcomes.
""")

# Model Accuracy Metrics - Default way. Both are displayed
# st.metric(label="Logistic Regression accuracy", value="98.25%", delta="Up 1.25% from previous model")
# st.metric(label="KNN accuracy", value="96.49%", delta="Down 1.75% from Logistic Regression")

# Model selection
model_choice = st.selectbox("Choose a model for prediction", ("Logistic Regression", "KNN"))
model = model_log_reg if model_choice == "Logistic Regression" else model_knn

# Display model-specific accuracy metric - Display only the one selected.
if model_choice == "Logistic Regression":
    st.metric(label="Logistic Regression accuracy", value="98.25%", delta="Up 1.25% from previous model")
elif model_choice == "KNN":
    st.metric(label="KNN accuracy", value="96.49%", delta="Down 1.75% from Logistic Regression")

# Taking input from user with adjusted ranges for each slider
radius_mean = st.slider("Enter the radius mean", 6.0, 30.0, 0.1)
perimeter_mean = st.slider("Enter the perimeter mean", 40.0, 200.0, 0.5)
area_mean = st.slider("Enter the area mean", 100.0, 2500.0, 5.0)
concave_points_mean = st.slider("Enter the concave points mean", 0.0, 0.2, 0.001)
radius_worst = st.slider("Enter the radius worst", 7.0, 40.0, 0.5)
perimeter_worst = st.slider("Enter the perimeter worst", 50.0, 250.0, 0.5)
area_worst = st.slider("Enter the area worst", 200.0, 5000.0, 10.0)
concave_points_worst = st.slider("Enter the concave points worst", 0.0, 0.3, 0.001)

# Scaling function
def scale_input(input_data):
    # Using zip to directly pair each feature value with its mean and std_dev
    return [(value - mean[key]) / std_dev[key] for value, key in zip(input_data, mean.keys())]

# Prediction
scaled_data = scale_input([radius_mean, perimeter_mean, area_mean, concave_points_mean, 
                           radius_worst, perimeter_worst, area_worst, concave_points_worst])
pred = model.predict([scaled_data])
result = "Benign" if pred[0] == 0 else "Malignant"

# # Output - Default Streamlit
# if st.button("Predict"):
#     st.success(f"The predicted breast cancer type is {result}")

# Output - Custom HTML in Streamlit
if st.button("Predict"):
    if result == "Benign":
        st.markdown(f'<p style="color:green; font-size:24px;">The predicted breast cancer type is {result}</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p style="color:red; font-size:24px;">The predicted breast cancer type is {result}</p>', unsafe_allow_html=True)
