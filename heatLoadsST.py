import streamlit as st
from heatLoads.model import * 
from heatLoads.data import * 


st.title("Heat Loads evaluations")

st.markdown(
    r"""This app evaluates the summer Heat Loads for a multizone building.   
    The data are contained in an Excel file including 4 sheets: 
    """)
st.markdown(r"""
    __zones__ : Number of occupants and walls areas of the zones   
    __walls__ : Parameters of the wall types   
    __windows__ : Parameters of the windows and solar screens   
    __general__ : General building data 
    
    """)

st.image("Fig.jpg", width=400)


fl = None
dt = None
dtname = None

st.markdown(r"""__You can:__""")

st.markdown(r"""
        - Simulate standard example (takes a few minutes)   
        - Download template data file  
        - Upload a data file (based on template) and simulate      
    """)

choices = ('Choose','Simulate standard example','Download template data file', 'Upload a data file and simulate')
option = st.selectbox('Choose:', choices)

if option == choices[1]:
    dt = "Heat_loads_data.xlsx"
    dtname = dt

if option == choices[2]:
    lnk = "[Download template](https://github.com/maajdl/glpkModelXLL/raw/master/Stigler's%201939%20diet%20problem%20v1.xlsx)"
    st.markdown(lnk)

if option == choices[3]:
    fl = st.file_uploader("Drop data excel file:")
    if fl is not None:
        dt = fl.getvalue()
        dtname = fl.name        
        
if dt is not None:
    st.title(f"""Heat loads [W]""")
    st.header(f"""for {dtname}""")
    dzo = pd.read_excel(dt, sheet_name="zones")
    dwl = pd.read_excel(dt, sheet_name='walls')
    dwd = pd.read_excel(dt, sheet_name='windows')
    dwg = pd.read_excel(dt, sheet_name='general')
    
    results = simulation(dzo, dwl, dwd, dwg)
    st.write(results.T)

    st.subheader("All Zones Maximum Heat Load : " + str(results["Total (W)"].max()) + " [W]")
    st.line_chart(results["Total (W)"])

    for zone in list(results.columns)[:-1]:
        st.subheader(zone[:-3] + "Maximum Heat Load : " + str(results[zone].max()) + " [W]")
        st.line_chart(results[zone])


