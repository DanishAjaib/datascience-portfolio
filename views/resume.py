import streamlit as st
from PIL import Image
import streamlit_antd_components as sac
import streamlit.components.v1 as components





#####################
# Header 


image = Image.open('dp2.png')
st.image(image, width=150)

st.markdown('## Summary', unsafe_allow_html=True)
st.info('''
Versatile individual with 10+ years of programming experience as a hobbyist and a freelance developer with further experience in Exploratory Data Analysis, Data Visualization, Hypothesis testing and Machine Learning through personal and university projects. 
''')


#####################
# Custom function for printing text
def txt(a, b):
  col1, col2 = st.columns([4,1])
  with col1:
    st.markdown(a)
  with col2:
    st.markdown(b)

def txt2(a, b):
  col1, col2 = st.columns([1,4])
  with col1:
    st.markdown(f'`{a}`')
  with col2:
    st.markdown(b)

def txt3(a, b):
  col1, col2 = st.columns([1,2])
  with col1:
    st.markdown(a)
  with col2:
    st.markdown(b)
  
def txt4(a, b, c):
  col1, col2, col3 = st.columns([1.5,2,2])
  with col1:
    st.markdown(f'`{a}`')
  with col2:
    st.markdown(b)
  with col3:
    st.markdown(c)


def txt5(a, b):
  col1, col2 = st.columns([1,9])
  with col1:
    st.markdown(a)
  with col2:
    st.markdown(b)

def text_project(title, link_txt, description, skills, link):
  txt(f'#### {title}','')
  st.markdown(f'''
  {description}
  ''')
  txt5('**Skills**', ", ".join(f"`{skill}`" for skill in skills))
  st.link_button(label=f'{link_txt}', url=f'{link}')

st.markdown('''
## Work Experience
''')

txt('**Python/Flutter Developer, Space Shuttle Parking**, Sydney, Australia',
'2021-2023')
st.markdown('''
- Developed a backend server for a car park system using Python, Elastic Search, and Firestore,
  focusing on automating email monitoring and data processing/cleaning. 
- Integrated AI-driven functionalities, including an OpenAI GPT assistant for customer support.
- Created a Python script to monitor entry and exit camera streams, enabling automated gate operations for authorized
vehicles.
- Developed Flutter Apps for Web, Mobile, Windows and Tablets.
''')

txt('**Web Administrator, PossumPiper**, , Sydney , Australia',
'2012-2021')
st.markdown('''
- Created and managed WordPress websites.
- Analyzed website data to optimize experience and content.
''')

#####################
st.markdown('''
## Skills
''')
txt3('Programming', '`Python`, `R`, `Dart`, `Kotlin`, `Java`')
txt3('Data processing/wrangling', '`SQL`, `pandas`, `numpy`')
txt3('Data visualization', '`matplotlib`, `seaborn`, `plotly` , `Tableau`')
txt3('Machine Learning', '`scikit-learn`')
txt3('Model deployment', '`streamlit`,`Heroku`, `AWS`, ')

st.header('Projects')
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    '**Data Analysis & ML**', 
    '**Data Visualization**',
    '**Python Scripting**',
    '**Research & Writing**', 
    '**App Development**',  

])


with tab1:

  txt('#### Predictive modelling of British Airways customer bookings','')
  st.markdown('''
  Analysing customer bookings data and training a model to predict whether a customer completes a booking or not.
  ''')
  txt5('**Skills**', '`Data Cleaning`, `Data Visualization`, `Machine Learning`')
  st.link_button(label='Notebook', url='https://github.com/DanishAjaib/british_airways_internship_analysis/blob/main/main.ipynb')

  txt('#### British Airways Customer Review Analysis','')
  st.markdown('''
  Analysing customer reviews for British Airways using NLTK.
  ''')
  txt5('**Skills**', '`Web Scraping`, `Data Cleaning`, `NLP`, `Data Visualization`')
  st.link_button(label='Notebook', url='https://github.com/DanishAjaib/british_airways_internship_analysis/blob/main/main.ipynb')

  txt('#### Predicting Employee Churn','')
  st.markdown('''
  Analysing employee churn data and building a model to predict employee churn..
  ''')
  txt5('**Skills**', '`Data Wrangling`, `Data Cleaning`, `Machine Learning`, `Data Visualization`')
  st.link_button(label='Notebook', url='https://github.com/DanishAjaib/employee_churn')

  txt('#### Tweet Analysis and Clustering','')
  st.markdown('''
  A project demonstrating Data Cleaning, Enrichment, Analysis, Topic Extraction, Classification and Clustering of a twitter dataset.
  ''')
  txt5('**Skills**', '`Data Wrangling`, `Data Cleaning`, `Machine Learning`, `Data Visualization`')
  st.link_button(label='Notebook', url='https://github.com/DanishAjaib/Tweet-Analysis-and-Clustering')

  txt('#### Australia Rain Visualization and Forecast','')
  st.markdown('''
  A hoppy project analysing Australia rain patterns and building a model for predicting next day rain.
  ''')
  txt5('**Skills**', '`Pandas`, `Python`, `Web Scraping`, `EDA`, `Model Building`, `Data Visualization`')
  st.link_button(label='Notebook', url='https://github.com/DanishAjaib/australia_rain_forecast')

  txt('#### Amazon Sentiment Analysis','')
  st.markdown('''
  A university project focusing on Web Scraping, Data Cleaning, Data Analsis and finally building a Model for sentiment prediction.
  ''')
  txt5('**Skills**', '`Pandas`, `Python`, `Web Scraping`, `EDA`, `Model Building`, `Data Visualization`')
  st.link_button(label='Notebook', url='https://github.com/DanishAjaib/Amazon-Sentiment-Analysis')

with tab2:

  txt('#### Sales/Profit Performance Dashboard','')
  st.markdown('''
    A Tableau dashboard visualizing Sales/Profit performance.
  ''')
  txt5('**Skills**', '`Tableau`, `Data Wrangling`, `Data Visualization`')
  st.link_button(label='Dashboard', url='https://public.tableau.com/app/profile/danish.ajaib/viz/SalesProfitPerformance_17206679136970')

  txt('#### Email Campaign Dashboard','')
  st.markdown('''
    A Tableau dashboard visualizing an email marketing campaign.
  ''')
  txt5('**Skills**', '`Tableau`, `Data Visualization`')
  st.link_button(label='Dashboard', url='https://public.tableau.com/app/profile/danish.ajaib/viz/EmailCampaignDashboard_17140166688400/Dashboard1')

  txt('#### Seoul Bicycle Rental Visualization','')
  st.markdown('''
    A Tableau dashboard visualizing an email marketing campaign.
  ''')
  txt5('**Skills**', '`Tableau`, `Data Visualization`')
  st.link_button(label='Dashboard', url='https://public.tableau.com/app/profile/danish.ajaib/viz/SeoulBicycleRental_17064119275750/Dashboard1')


with tab3:
  txt('#### Backend App','')
  st.markdown('''
   This script is responsible for 
   - Monitoring email inboxes using Microsoft Graph API.
   - Data Ingestion, cleaning and preparation.
   - Pushing the  data to Elastic Search and Firestore.
   - Monitoring customer  email inqueries and responding to them using Open AI ChatGPT Assisstants API.
  ''')
  txt5('**Skills**', '`Python`, `Data Cleaning`, `Gen AI`, `Heroku`, `Elastic Search`, `Firestore`, `Redis`')
  st.link_button(label='Source Code', url='https://gitfront.io/r/danishajaib/YFg4zJVy5TXo/email-server/')

  txt('#### Python Script for Monitoring Car Park Cameras','')
  st.markdown('''
   This script connects to multiple camera streams responsbile Automatic Number Plate Recognition. Based on the authorization status of a vehicle entering or exiting, the script controls gate operations by sending a singal to an onsite microcontroller. 
  ''')
  txt5('**Skills**', '`Python`, `API`, `Heroku`,')


with tab4:
  txt('#### A Survey of Cross-Domain Recommender Systems','')
  st.markdown('''
  This work is a literature review and delves into overcoming traditional collaborative filtering challenges using advanced techniques like transfer learning and neural collaborative filtering. It highlights the transformative potential of these systems across diverse industries, from e-commerce to healthcare. My journey through this project enhanced my understanding of data science intricacies and future tech trends, contributing valuable insights into the evolution of user experience in digital domains.
  ''')
  txt5('**Skills**', '`Technial Writing`, `Research Skills`, `Analytical Skills`, `Communication`, `Synthesis of Information`')
  st.link_button(label='PDF', url='https://www.linkedin.com/in/danish-ajaib-865528107/details/projects/1704617200423/single-media-viewer?type=DOCUMENT&profileId=ACoAABrzyM0BjMYrqY8KqDQ1UnHu_HgkagWHQ6A&lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_projects_details%3BDDyIlEzdTFGKgFdJDz3dRw%3D%3D')

  txt('#### Market Analysis and Strategy','')
  st.markdown('''
  In this project during my internship, I was part of a team of four tasked with developing a market analysis and marketing plan for an emerging oat milk brand. We conducted in-depth research on market trends and competitor strategies, focusing on plant-based dairy alternatives. Our approach included frequent collaborative team meetings and strategic discussions with the CEO, leading to a tailored marketing strategy. The strategy emphasized health, sustainability, and allergen-free benefits, targeting diverse consumer demographics. This project honed my skills in teamwork, strategic planning, and high-level corporate communication.
  ''')
  txt5('**Skills**', '`Communication`, `Market Data Analysis`, `Market Research`, `Team Collaboration`')
  st.link_button(label='Report', url='https://www.linkedin.com/in/danish-ajaib-865528107/details/projects/1704617545125/single-media-viewer?type=DOCUMENT&profileId=ACoAABrzyM0BjMYrqY8KqDQ1UnHu_HgkagWHQ6A&lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_projects_details%3BDDyIlEzdTFGKgFdJDz3dRw%3D%3D')

with tab5:
  txt('#### Budget Management App','')
  st.markdown('''
   A budget management app written in Java and Kotlin.
  ''')
  txt5('**Skills**', '`Dart`, `Java`,')
  st.link_button(label='Source Code', url='https://gitfront.io/r/danishajaib/CipWDJRcygiL/SpesaBudgetManager/')



 
#####################
st.markdown('''
## Licenses & Certification
''')
txt4('Coursera', 'Google Advanced Data Analytics Specialization', '[See Certificate](https://www.coursera.org/account/accomplishments/specialization/LLZCWPL23U6R)')
txt4('Coursera', 'Build and Operate Machine Learning Solutions with Azure', '[See Certificate](https://www.coursera.org/share/https://coursera.org/share/521bc1dce8b6ffb8b8ab6c0c8c90de13)')
txt4('Coursera', 'Create Machine Learning Models with Microsoft Azure', '[See Certificate](https://www.coursera.org/account/accomplishments/certificate/2PQ8FW5W7YFF)')
txt4('Coursera', 'Microsoft Azure Machine Learning for Data Scientists', '[See Certificate](https://coursera.org/share/695c787822b9ec8f7b0ed35a1a499f77)')
txt4('Coursera', 'Time Management Fundamentals', '[See Certificate](https://www.linkedin.com/learning/certificates/d22bec6e4e60f5564ce73f486b70c7c6deae364d636362a2e15b59a0468f23c0)')
txt4('Macquarie University', 'Study Australia Industry Experience Program Completer','[See Certificate](https://api.practera.com/achieve/user_achievements/assertion/d4r7junAW.json)')
txt4('Coursera', 'Business Analysis Foundations','[See Certificate](https://www.linkedin.com/learning/certificates/2bf0d93c2b32bbdf6a380f7baf0daca611c8be65bfe6015bea5638c592d466de?trk=share_certificate)')
txt4('Macquarie University', 'Study Australia Industry Experience Program Completer','[See Certificate](https://api.practera.com/achieve/user_achievements/assertion/d4r7junAW.json)')




#####################
st.markdown('''
## Education
''')

txt('**Masters of Data Science**, *Macquarie University*, Australia',
'2019-2023')
st.markdown('''
**Date Conferred:** February 29, 2024  
**Date Qualified:** December 7, 2023  
            
**Key Courses:**
- Web Technology
- Data Science
- Introductory Statistics
- Machine Learning
- Big Data
- Big Data Technologies
- Mining Unstructured Data
- Applications of Data Science
- Statistical Graphics
- Statistical Inference
- Generalized Linear Models
- Multivariate Analysis
- Time Series
- Modern Computational Statistical Methods
''')

txt('**Bachelors of  Computer Science**, *Iqra University*, Pakistan',
'2014-2018')
# st.markdown('''
# - GPA: `3.65`
# - Graduated with First Class Honors.
# ''')

#####################

#####################
st.markdown('''
## Profiles
''')
txt2('LinkedIn', 'https://www.linkedin.com/in/danish-ajaib-865528107/')
txt2('Wakatime', 'https://wakatime.com/@danishajaib')
txt2('Tableau', 'https://public.tableau.com/app/profile/danish.ajaib/vizzes')
txt2('Medium', 'https://medium.com/@danishajaib93')
