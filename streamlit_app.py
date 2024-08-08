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
)

british_airways_bookings_analysis_page = st.Page(
  page='views/british_airways_bookings_analysis.py',
  title='British Airways Bookings Analysis',
)

australia_rain_analysis_page = st.Page(
  page='views/australia_rain_forecast.py',
  title='Australia next day rain analysis',
)
pg = st.navigation(
  {
    'Info': [resume_page],
    'Notebooks': [british_airways_analysis_page,british_airways_bookings_analysis_page,  australia_rain_analysis_page]
  }
)
st.logo('assets/codingisfun_logo.png')
st.sidebar.text('Made with ❤ in Sydney')


pg.run()





