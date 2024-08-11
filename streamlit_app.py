import streamlit as st
from PIL import Image
import streamlit_antd_components as sac
import streamlit.components.v1 as components



resume_page = st.Page(
  page='views/resume.py',
  title='Resume',
  icon=':material/account_circle:',
  default=True
)

british_airways_analysis_page = st.Page(
  page='views/british_airways_reviews_notebook.py',
  title='British Airways Customer Reviews Analysis',
  icon=':material/description:',
)

british_airways_bookings_analysis_page = st.Page(
  page='views/british_airways_bookings_analysis.py',
  title='British Airways Bookings Analysis',
  icon=':material/description:',
)

accenture_internship = st.Page(
  page='views/accenture_data_analytics_internship.py',
  title='Accenture Data Analytics and Visualizations Internship',
  icon=':material/description:',
)
pg = st.navigation(
  {
    'Info': [resume_page],
    'Notebooks': [british_airways_analysis_page,british_airways_bookings_analysis_page,  accenture_internship]
  }
)
st.logo('assets/codingisfun_logo.png')
st.sidebar.text('Made with ❤ in Sydney')

pg.run()





