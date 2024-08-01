
import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from matplotlib import pyplot as plt
import stopwords
from textblob import TextBlob
import emoji
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import seaborn
import altair as alt
import streamlit as st
import pandas as pd
import altair as alt
import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os.path


st.title('British Airways customer reviews analysis')
st.text('In this project we will analyse a dataset of 10K customer reviews on British Airlines. To start off, we scrap 10000 revies off airlinequality.com for British Airlines.')

##
# nltk.download()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

df = pd.DataFrame(columns=['reviews'])

@st.cache_resource
def get_sentiment(text):
    blob = TextBlob(text)
    # Get the polarity score
    polarity = blob.sentiment.polarity
    # Determine the sentiment based on the polarity score
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'

@st.cache_data(persist='disk')
def preprocess_text(text):
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

    
def contains_emoji(text):
    # check if the text contains any emojis
    return any(char in emoji.UNICODE_EMOJI['en'] for char in text)

@st.cache_data(persist='disk')
def clean_review(text):
    import re
    # remove "VERIFIED" and "NOT VERIFIED" (case insensitive)
    text = re.sub(r'(verified|not verified)', '', text, flags=re.IGNORECASE)
    # remove the "|" character
    text = re.sub(r'\|', '', text)
    # remove mentions, special characters, hashtags, and URLs
    text = re.sub(r"(@[A-Za-z0-9_]+)|[^\w\s]|#|http\S+", "", text)
    # remove any extra spaces that might result from the removal
    text = ' '.join(text.split())
    return text

@st.cache_data(persist='disk')
def scrap_reviews():
    base_url = "https://www.airlinequality.com/airline-reviews/british-airways"
    pages = 100
    page_size = 100

    reviews = []

    # for i in range(1, pages + 1):
    for i in range(1, pages + 1):

        print(f"Scraping page {i}")

        # Create URL to collect links from paginated data
        url = f"{base_url}/page/{i}/?sortby=post_date%3ADesc&pagesize={page_size}"

        # Collect HTML data from this page
        response = requests.get(url)

        # Parse content
        content = response.content
        parsed_content = BeautifulSoup(content, 'html.parser')
        for para in parsed_content.find_all("div", {"class": "text_content"}):
            reviews.append(para.get_text())
        
        print(f"   ---> {len(reviews)} total reviews")

@st.cache_data(persist='disk')
def generate_wordcloud(reviews, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(reviews)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    
    # Save to a BytesIO object
    image_file = f"{title}.png"
    plt.savefig(image_file, format='png')
    plt.close()
    return image_file


@st.cache_data(persist='disk')
def prepare_data():
    prepared = pd.read_csv('data/reviews.csv')
    prepared.drop(df.filter(regex="Unnamed"),axis=1, inplace=True)
    prepared['reviews_clean'] = prepared.apply(lambda x:clean_review(x['reviews'].lower()), axis=1)
    prepared['preprocessed_review'] = prepared['reviews_clean'].apply(preprocess_text)
    prepared['sentiment'] = prepared['preprocessed_review'].apply(lambda x: get_sentiment(x))
    prepared.to_csv('data/reviews_clean.csv')

    return prepared

def get_data():
    if not os.path.isfile('data/reviews_clean.csv'):
        return prepare_data()
    else:
        df = pd.read_csv('data/reviews_clean.csv', index_col=0)
        df.drop(df.filter(regex="Unnamed"),axis=1, inplace=True)
        return df

df = get_data()
df.drop(df.filter(regex="Unnamed"),axis=1, inplace=True)
st.dataframe(df.head(), width=920)
st.markdown('A first look of the dataset tells us that it only has one column that contains the customer reviews about the airline. Next we need to check if the dataset contains any duplicates or missing values.')
st.markdown('# Data Cleaning and Preparation')
st.markdown('''
Next, we prepare the data using the following steps:
 - Clean the reviews by removing the 'VERIFIED | NOT VERIFIED' text and any special characters it may contain.
 - Create a new column containing the tokens of each review. Tokens in NLP are basically individual words without any stopwords. Tokens are useful for text computing word frequencies, identifying key terms, part-speech tagging, named entity recognition, sentiment analysis and more.
 - Create a new column containing the sentiment of each review.
''')

st.dataframe(df.head(), width=920)
palette_color = seaborn.color_palette('bright') 
# Calculate sentiment counts
counts = df['sentiment'].value_counts().reset_index()
counts.columns = ['sentiment', 'count']

# Calculate percentages
counts['percentage'] = (counts['count'] / counts['count'].sum()) * 100

# Add labels for displaying both count and percentage
counts['label'] = counts.apply(lambda row: f"{row['sentiment']} ({row['count']})", axis=1)
color_red, color_neutral, color_green = '#FF204E', '#9DB2BF', '#BED754'
# Define colors
palette_color = [color_red, color_neutral, color_green]

# Create Altair chart
donut_chart = alt.Chart(counts).mark_arc(innerRadius=50).encode(
    theta=alt.Theta(field='count', type='quantitative', stack=True),
    color=alt.Color(field='sentiment', type='nominal', scale=alt.Scale(range=palette_color)),
    tooltip=[alt.Tooltip('sentiment', title='Sentiment'),
             alt.Tooltip('count', title='Count'),
             alt.Tooltip('percentage', title='Percentage', format='.1f')]
).properties(
    title='Sentiment Analysis',
    width=400,
    height=400
)

# Display the chart in Streamlit
##st.altair_chart(donut_chart, use_container_width=True)


sentiment_counts = df['sentiment'].value_counts().reset_index()

# Define colors

# Create Altair bar chart
bar_chart = alt.Chart(sentiment_counts).mark_bar().encode(
    x=alt.X('sentiment:N', title='Sentiment'),
    y=alt.Y('count:Q', title='Number of Reviews'),
    color=alt.Color('sentiment:N', scale=alt.Scale(range=palette_color)),
    tooltip=[alt.Tooltip('sentiment', title='Sentiment'),
             alt.Tooltip('count', title='Number of Reviews')]
).properties(
    title='Sentiment Distribution',
    width=600,
    height=400
)

# Display the chart in Streamlit
##st.altair_chart(bar_chart, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.altair_chart(bar_chart, use_container_width=True)

with col2:
    st.altair_chart(donut_chart, use_container_width=True)

st.markdown('''From the above charts, we can see that the majority of British Airline customers 
            have happy while 29.2% areunhappy with BA services. Only 0.7% are neutral. Next, we look 
            at the overall sentiment distribution. Next, weplot the word clouds for both positive and negative reviews.''')







# Generate word clouds
positive_reviews = ' '.join(df[df['sentiment'] == 'positive']['preprocessed_review'])
negative_reviews = ' '.join(df[df['sentiment'] == 'negative']['preprocessed_review'])
with st.spinner('Generating Wordclouds'):

    # Create word clouds
    positive_wordcloud_file = generate_wordcloud(positive_reviews, 'Positive Reviews Word Cloud')
    negative_wordcloud_file = generate_wordcloud(negative_reviews, 'Negative Reviews Word Cloud')

# Display in Streamlit
st.title("Customer Reviews Word Clouds")

# Load and display positive word cloud
with open(positive_wordcloud_file, "rb") as img_file:
    st.image(img_file.read(), caption='Positive Reviews Word Cloud')

# Load and display negative word cloud
with open(negative_wordcloud_file, "rb") as img_file:
    st.image(img_file.read(), caption='Negative Reviews Word Cloud')

st.markdown('''
From the above word clouds , we can infer that both positive and negative reviewws frequently mention `flight`, `seat` and `service`, indicating these are critical aspects of the passenger experience that can greatly influence overall satistication. Positive reviews are mostly characterized by words like `good`, `great`, `comfortable`, and `excellent`, while the negative reviews frequently use words like `terrible`, `awful`, `poor` and `bad`. Negative reviews specifically highlight issued related to `delays`, `luggage` and `customer service`, while positive reviews emphasize `good service`, `comfortable seating` and pleasant `staff interactions`.
''')

st.title('Topic Extraction')
st.markdown('We can do further analysis by extracting topics for both positive and negative reviews.')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

@st.cache_data
def get_top_topics(sentiment, topic_count):
    all_topics = []
    # vectorize the preprocessed reviews
    vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(df[df['sentiment'] == sentiment]['preprocessed_review'])

    # LDA model
    lda = LatentDirichletAllocation(n_components=topic_count, random_state=0)
    lda.fit(X)
    topics_df = pd.DataFrame(
        
    )
    # display the top words in each topic
    for i, topic in enumerate(lda.components_):
        topics_df[f'Topic {i}'] = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]

        print(f"Top words for topic #{i}:")
        print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
        print("\n")
        all_topics.append(f'{sentiment.capitalize()} Topic {i}: {[vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]}')
        
    return all_topics

@st.cache_data
def get_bigrams():
    # Extract bigrams for positive reviews
    vectorizer_pos = CountVectorizer(ngram_range=(2, 2), stop_words='english')
    X_pos = vectorizer_pos.fit_transform(df[df['sentiment'] == 'positive']['preprocessed_review'])
    bigrams_pos = vectorizer_pos.get_feature_names_out()
    bigram_counts_pos = X_pos.toarray().sum(axis=0)
    bigram_freq_pos = dict(zip(bigrams_pos, bigram_counts_pos))
    sorted_bigrams_pos = sorted(bigram_freq_pos.items(), key=lambda x: x[1], reverse=True)

    # Extract bigrams for negative reviews
    vectorizer_neg = CountVectorizer(ngram_range=(2, 2), stop_words='english')
    X_neg = vectorizer_neg.fit_transform(df[df['sentiment'] == 'negative']['preprocessed_review'])
    bigrams_neg = vectorizer_neg.get_feature_names_out()
    bigram_counts_neg = X_neg.toarray().sum(axis=0)
    bigram_freq_neg = dict(zip(bigrams_neg, bigram_counts_neg))
    sorted_bigrams_neg = sorted(bigram_freq_neg.items(), key=lambda x: x[1], reverse=True)

    # Create a DataFrame for the top 5 bigrams from both positive and negative reviews
    top_bigrams_pos = sorted_bigrams_pos[:5]
    top_bigrams_neg = sorted_bigrams_neg[:5]

    # Combine the data into a DataFrame
    combined_bigrams = pd.DataFrame({
        'bigram': [bigram[0] for bigram in top_bigrams_pos] + [bigram[0] for bigram in top_bigrams_neg],
        'frequency': [bigram[1] for bigram in top_bigrams_pos] + [bigram[1] for bigram in top_bigrams_neg],
        'sentiment': ['positive'] * len(top_bigrams_pos) + ['negative'] * len(top_bigrams_neg)
    })

    # Pivot the DataFrame for plotting
    # pivot_df = combined_bigrams.pivot(index='bigram', columns='sentiment', values='frequency').fillna(0)

    return combined_bigrams

@st.cache_data
def create_bigrams_chart(combined_bigrams):
     # Create Altair bar chart
    return alt.Chart(combined_bigrams).mark_bar().encode(
        x=alt.X('bigram:N', sort=None, axis=alt.Axis(title='Bigrams')),
        y=alt.Y('frequency:Q', axis=alt.Axis(title='Frequencies')),
        color=alt.Color('sentiment:N', scale=alt.Scale(domain=['positive', 'negative'], range=[color_green, color_red])),
        tooltip=['bigram', 'frequency', 'sentiment']
    ).properties(
        title='Top 5 Bigrams in Positive and Negative Reviews',
        width=700,
        height=400
    ).configure_axis(
        labelAngle=-45,
        labelAlign='right'
    ).configure_title(
        fontSize=20,
        anchor='middle'
    )

@st.cache_data
def create_boxplot():
    # Calculate review length
    df['review_length'] = df['preprocessed_review'].apply(lambda x: len(x.split()))

    # Create Altair boxplot
    return alt.Chart(df).mark_boxplot(extent='min-max').encode(
        x=alt.X('sentiment:N', title='Sentiment'),
        y=alt.Y('review_length:Q', title='Review Length'),
        color=alt.Color('sentiment:N', scale=alt.Scale(domain=['positive', 'negative'], range=[color_green, color_red]))
    ).properties(
        title='Review Length by Sentiment',
        width=600,
        height=400
    ).configure_axis(
        labelAngle=0
    ).configure_title(
        fontSize=20,
        anchor='middle'
    )
topics_pos = get_top_topics(sentiment='positive', topic_count=3)
topics_neg = get_top_topics(sentiment='negative', topic_count=3)

st.text('Positive Topics')
for topic in topics_pos:
    st.text(topic)
st.text('Negative Topics')
for topic in topics_neg:
    st.text(topic)

st.markdown(''' 
Both positive and negative reviews seem to contain simillar topics which suggests that the differing experiences relate to simillar aspects of British Airline services. This means that aspecs like `Seat and Cabin Comfort`, `Service Quality`, `Food and Meals`, `Flight Experience and Timing`, and  `Economy and Business classes` are the main areas that impact the overall passenger experience. British Airways should focus on consistently improving these critical aspects. By addressing the negative feedback and maintaining the positive aspects, the airline can work towards providing a more uniform and high quality experience for all passengers.

Next, for further analysis, we will extract the top 10 most common bigrams(2 words) for both positive and negative reviews.           
''')

with st.spinner('Getting Bigrams'):
    # Pivot the DataFrame for plotting
    combined_bigrams = get_bigrams()
    # Display in Streamlit
    st.title("Top 5 Bigrams in Positive and Negative Reviews")
    st.altair_chart(create_bigrams_chart(combined_bigrams=combined_bigrams), use_container_width=True)

st.markdown('''
The bigram frequency distribution chart shows us the top aspects of Birtish Airways that contribute to the
majority of positive and negative experiences. This could simply mean that customers on average have a
positive experience but some customer experiences seem to negative about the simillar aspects of the travel experience. 
''')

# Display in Streamlit
st.title("Review Length by Sentiment")
st.altair_chart(create_boxplot(), use_container_width=True)

st.markdown('''
Finally, the box plot of review length shows that that on average negative and positive reviews are simillar in 
length ranging from ~50 to a little over 100 words with positive reviews being slightly larger. 
Neutral reviews seem to be very short on average containing about 50 words.
''')
from streamlit.components.v1 import html
def open_source_code(url):
    open_script= """
        <script type="text/javascript">
            window.open('%s', '_blank').focus();
        </script>
    """ % (url)
    html(open_script)
st.button(label='Source Code', use_container_width=True, on_click=open_source_code('https://github.com/DanishAjaib/british_airways_internship_analysis/blob/main/main.ipynb'))