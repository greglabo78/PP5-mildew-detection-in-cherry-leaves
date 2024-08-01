import streamlit as st 
import matplotlib.pyplot as plt 


def page_project_hypothesis_body():
    st.write("### Project Hypothesis and Validation")

    st.success(
        f"* We believe that cherry leaves affected by powdery mildew exhibit distinct symptoms "
        f"Initially, a light-green, circular lesion appears on either surface of the leaf \n\n"
        f"followed by the development of a subtutle, white cotton-like growth in the affected area \n\n"
        f"* An Image Montage highlights the noticeable difference between healthy and infected leaves. "
        f" However, average Image, Variability Image and comparisons of average differences studies did not show "
        f"any distinct pattern to reliably differentiate between healthy and infected leaves."
    )

    