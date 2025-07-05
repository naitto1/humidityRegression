import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Page Config
st.set_page_config(
    page_title="Weather Analysis Dashboard"
    layout="wide"
)

# Load data
@st.cache_data
def train_model(df):
    features = ["Temperature (C)", "Humidity", "Wind Speed (km/h)", "Wind Bearing (degrees)", "Visibility (km)", "Pressure (millibars)"]
    target = "Apparent Temperature (C)"
    X = df[features]
    y = df[target]
    model = LinearRegression()
    model.fit(X, y)
    return model, features

if df is not None:
    model, feature_names = train_model(df)

    # Sidebar for user input
    st.sidebar.header("Predict Apparent Temperature")
    st.sidebar.markdown("Use sidebars to set weather")

    input_temp = st.sidebar.slider("Temperature (°C)", float(df['Temperature (C)'].min()), float(df['Temperature (C)'].max()), float(df['Temperature (C)'].mean()))
    input_humidity = st.sidebar.slider("Humidity", float(df['Humidity'].min()), float(df['Humidity'].max()), float(df['Humidity'].mean()))
    input_wind_speed = st.sidebar.slider("Wind Speed (km/h)", float(df['Wind Speed (km/h)'].min()), float(df['Wind Speed (km/h)'].max()), float(df['Wind Speed (km/h)'].mean()))
    input_wind_bearing = st.sidebar.slider("Wind Bearing (degrees)", 0, 359, 180)
    input_visibility = st.sidebar.slider("Visibility (km)", float(df['Visibility (km)'].min()), float(df['Visibility (km)'].max()), float(df['Visibility (km)'].mean()))
    input_pressure = st.sidebar.slider("Pressure (millibars)", float(df['Pressure (millibars)'].min()), float(df['Pressure (millibars)'].max()), float(df['Pressure (millibars)'].mean()))

    # --- Prediction ---
    input_data = pd.DataFrame([[input_temp, input_humidity, input_wind_speed, input_wind_bearing, input_visibility, input_pressure]], columns=feature_names)
    prediction = model.predict(input_data)[0]

    st.sidebar.subheader("Predicted Apparent Temperature:")
    st.sidebar.metric(label="Apparent Temp (°C)", value=f"{prediction:.2f}°C")


    # --- Main Page Content ---
    st.title("🌤️ Weather History Analysis and Prediction")
    st.markdown("This dashboard explores the relationship between weather conditions and provides a tool to predict the apparent temperature.")

    # --- EDA Section ---
    st.header("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Average Temperatures")
        df_monthly = df.set_index('Formatted Date')[['Temperature (C)', 'Apparent Temperature (C)']].resample('M').mean()
        st.line_chart(df_monthly)
        st.markdown("We can observe a clear seasonal pattern in temperatures.")

    with col2:
        st.subheader("Humidity vs. Temperature")
        fig, ax = plt.subplots()
        sns.scatterplot(x='Temperature (C)', y='Humidity', data=df.sample(1000), ax=ax, alpha=0.5) # Sample for performance
        ax.set_title("Temperature vs. Humidity")
        st.pyplot(fig)
        st.markdown("There is a general trend of lower humidity at higher temperatures.")


    st.subheader("Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)
    st.markdown("The heatmap shows the strong positive correlation between Temperature and Apparent Temperature, and the negative correlation between Temperature and Humidity.")

    # --- Data Viewer ---
    st.header("Raw Data Viewer")
    st.markdown("Tick the checkbox to view the raw data.")
    if st.checkbox("Show Raw Data"):
        st.write(df)

else:
    st.info("Awaiting data file to begin analysis.")
