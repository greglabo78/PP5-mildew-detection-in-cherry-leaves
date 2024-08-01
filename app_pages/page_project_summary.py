import streamlit as st
import matplotlib.pyplot as plt 


def page_project_summary_body():

    st.info ( 
        f"**General Information**\n"
        f"* Powdery mildew is a fungal disease that affects cherry trees"
        f"caused by the parasite Posdosphaera clandestina.\n"
        f"* As the fungus spreads, it creates a layer of mildew composed of numerous spores on the leaf surface"
        f"Visual indictaors of infected leaves include :\n"
        f"* Light-green, circular lesions appearing on either side of the leaf"
        f"A delicate white, cotton-like growth that eventually develops in the infected areas on either side of the leaf"
        f"Several leaves, infected and healthy were picked up and examined "
        f"**Project Dataset**\n"
        f"* The available dataset contains 4208 files, 2104 images of healthy leaves"
        f"and 2104 images of infected leaves each photographed against a neutral background ")
    

    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/greglabo78/PP5-mildew-detection-in-cherry-leaves/blob/main/README.md.")
    
    st.success(
        f"The project has 2 business requirements:\n"
        f"* 1 - The client is interested in having a study to differentiate"
        f"a healthy from an infected leaf visually.\n"
        f"* 2 - The client is interested in telling whether a given leaf is infected with powdery mildew or healthy. "
        )
    
