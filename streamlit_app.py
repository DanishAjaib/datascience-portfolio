import streamlit as st
from PIL import Image

with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

#####################
# Header 
st.write('''
# Danish Ajaib
##### *Resume* 
''')

image = Image.open('dp.png')
st.image(image, width=150)

st.markdown('## Summary', unsafe_allow_html=True)
st.info('''
Versatile individual with 10+ years of programming experience as a hobbyist and a freelance developer with further experience in Exploratory Data Analysis, Data Visualization, Hypothesis testing and Machine Learning through personal and university projects. 
''')

#####################
# Navigation

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #16A2CB;">
  <a class="navbar-brand" href="https://youtube.com/dataprofessor" target="_blank">Chanin Nantasenamat</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item active">
        <a class="nav-link disabled" href="/">Home <span class="sr-only">(current)</span></a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#education">Education</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#work-experience">Work Experience</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#bioinformatics-tools">Bioinformatics Tools</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#social-media">Social Media</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)

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

st.markdown('''
## Projects
''')



#####################
st.markdown('''
## Licenses & Certification
''')
txt4('Coursera', 'Google Advanced Data Analytics Specialization', 'https://www.coursera.org/account/accomplishments/specialization/LLZCWPL23U6R')
txt4('Coursera', 'Build and Operate Machine Learning Solutions with Azure', 'https://www.coursera.org/share/https://coursera.org/share/521bc1dce8b6ffb8b8ab6c0c8c90de13')
txt4('Coursera', 'Create Machine Learning Models with Microsoft Azure', 'https://www.coursera.org/account/accomplishments/certificate/2PQ8FW5W7YFF')
txt4('Coursera', 'Microsoft Azure Machine Learning for Data Scientists', 'https://coursera.org/share/695c787822b9ec8f7b0ed35a1a499f77')
txt4('Coursera', 'Time Management Fundamentals', 'https://www.linkedin.com/learning/certificates/d22bec6e4e60f5564ce73f486b70c7c6deae364d636362a2e15b59a0468f23c0')
txt4('Macquarie University', 'Study Australia Industry Experience Program Completer','https://api.practera.com/achieve/user_achievements/assertion/d4r7junAW.json')
txt4('Coursera', 'Business Analysis Foundations','https://www.linkedin.com/learning/certificates/2bf0d93c2b32bbdf6a380f7baf0daca611c8be65bfe6015bea5638c592d466de?trk=share_certificate')
txt4('Macquarie University', 'Study Australia Industry Experience Program Completer','https://api.practera.com/achieve/user_achievements/assertion/d4r7junAW.json')




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
