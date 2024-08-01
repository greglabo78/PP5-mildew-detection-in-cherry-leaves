import streamlit as st
from app_pages.multipage import MultiPage

#load pages scripts
from app_pages.page_project_summary import page_project_summary_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body



app = MultiPage(app_name="Powdery Mildew Detector")  # Create an instance of the app


# Add your app pages here using .add_page()
app.add_page("Project Summary", page_project_summary_body)
app.add_page("Project Hypothesis", page_project_hypothesis_body)



app.run()  # Run the app