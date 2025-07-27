import streamlit as st
import pandas as pd
import numpy as np

st.title("Hello Everyone this is our first application!!")
st.write("This is SCIT in Pune")

df=pd.DataFrame({
    'first column' : [1,2,3,4],
    'second column' : [10,20,30,40]
})

st.write("Here is the dataframe")
st.write(df)

chart_data=pd.DataFrame(np.random.randn(20,3),columns=['a','b','c'])
st.bar_chart(chart_data)
