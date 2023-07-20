# -*- coding: utf-8 -*-


import numpy as np
import pickle
import streamlit as st

loaded_model=pickle.load(open('D:/internship/iris flower classification/trained_model.sav','rb'))

def Iris_predict(input_data):
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    if (prediction[0] == 0):
      return 'Iris setosa'
    elif (prediction[0] == 1):
      return 'Iris versicolor'
    else:
      return 'Iris virginica'

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://media.istockphoto.com/id/1419735746/photo/fall-background-with-vibrant-dahlias-on-wood-surface.webp?b=1&s=170667a&w=0&k=20&c=nRnXDj7ZKq3VDa4ud6n5_npPb8q70G2IF9PnlJ_msco=");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def main():
    
    
    add_bg_from_url()
    st.title('Iris Flower Classifier')
    
    #getting the input data
    
    SepalLength = st.text_input('Sepal length value')
    SepalWidth = st.text_input('Sepal width value')
    PetalLength = st.text_input('Petal length value')
    PetalWidth = st.text_input('Petal width value')
    
    result= ''
    
    if st.button('Predict type'):
        result= Iris_predict([SepalLength,SepalWidth,PetalLength,PetalWidth])
        
    st.success(result)
    
if __name__ == '__main__':
    main()
    
    