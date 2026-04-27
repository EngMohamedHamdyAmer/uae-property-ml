import streamlit as st
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

@st.cache_resource
def load_model():
    with open('model.pkl' , 'rb') as f :
        model = pickle.load(f)
    
    with open('model_columns.pkl' , 'rb') as f :
        columns = pickle.load(f)

    return model , columns

@st.cache_resource
def load_data():
    return pd.read_csv('Clean_data.csv')

model , model_columns = load_model()
df = load_data()

st.title( " 🏚️ UAE Property Price Prediction")
st.caption("Machine Learning model trained on 3,000+ scraped listings")

tab1 , tab2 = st.tabs([ 'Price Prediction' , 'Market Analysis'])

with tab1 :
    st.header( 'Predict Property Price')

    col1 , col2 = st.columns(2)

    with col1 :
        emirate = st.selectbox( 'Emirate' , sorted(df['Emirate'].dropna().unique()))
        beds = st.slider( 'Bedrooms' , 0 , 7 , 2)
        baths = st.slider( 'Bathrooms' , 1 , 8 , 2)

    with col2 :
        area = st.number_input( 'Area (Square Feet)' , 300 , 30000 , 1500 )
    

    if st.button( 'Predict Price' , use_container_width= True):

        if area <= 0:
            st.error("Area must be greater than 0")
            st.stop()

        
        input_dict = {
            'Beds' : beds,
            'Baths': baths,
            'Area' : area,
        }

        emirate_col = f"Emirate_{emirate}"
        if emirate_col in model_columns:
            input_dict[emirate_col] = 1

        input_df = pd.DataFrame( [input_dict] )
        input_df = input_df.reindex( columns= model_columns , fill_value= 0 )


        log_prediction = model.predict(input_df)[0]
        prediction = np.exp(log_prediction)

        st.success(f" Estimated Price: {prediction:,.0f} AED")
        st.info(f"≈ AED {prediction/1e6:.2f} Million")
        st.metric("Price per sqft", f"AED {prediction/area:.0f}")


with tab2:

        st.header('UAE Property Market Analysis')

        col1 , col2 , col3 = st.columns(3)

        col1.metric ( 'Total Number Of Listings' , f"{len(df)}")
        col2.metric ( 'Average price' , f"{df['Price'].mean()/1e6:0.2f}M AED")
        col3.metric ( 'Average Price per Sqft' , f"{df['Price per sqft'].mean():.2f} AED")

        st.divider()

        st.subheader( 'Medain Price By Emirate')
        fig1 , ax1 = plt.subplots(figsize= (10 , 4))
        df.groupby('Emirate')['Price'].median().sort_values().plot(kind= 'barh' , ax= ax1 , color= 'steelblue')
        ax1.xaxis.set_major_formatter( plt.FuncFormatter( lambda x ,_ : f'{x/1e6:.1f}M'))
        ax1.set_xlabel( 'Median Price (AED)')
        st.pyplot(fig1)

        st.divider()

        st.subheader( 'Average Price per Sqft By Emirate')
        fig2 , ax2 = plt.subplots(figsize= (10 , 4))
        df.groupby('Emirate')['Price per sqft'].mean().sort_values().plot(kind= 'barh' , ax= ax2 ,color= 'coral')
        ax2.set_xlabel( 'Average Price per Sqft (AED)')
        st.pyplot(fig2)

        st.divider()

        st.subheader( 'Property Type Distribution' )
        if 'Property type' in df.columns :
            fig3 , ax3 = plt.subplots(figsize= (10,4))
            df['Property type'].value_counts().plot(kind= 'bar' ,ax= ax3 , color= 'teal')

            ax3.set_xlabel( 'Property type')
            ax3.set_ylabel( 'Count')
            plt.xticks( rotation= 45)
            st.pyplot(fig3)

        st.divider()

        st.subheader('Key Findings')
        st.markdown("""
         -  **Best Model**: Random Forest + Log Transform — R² = 0.874
         -  **Dubai** costs **4x more** per sqft than Ajman
         -  **Area** is the strongest price predictor (correlation: 0.79)
         -  Adding **District** improved R² by 6.6%
         -  **Log Transform** reduced RMSE by 63,000 AED """)

        


        