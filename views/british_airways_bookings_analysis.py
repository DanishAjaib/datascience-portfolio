import streamlit as st
import pandas as pd
import pandas as pd
import altair as alt
import streamlit as st
import pandas as pd
import altair as alt
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
from sklearn.preprocessing import  StandardScaler
from scipy.stats import chi2_contingency


@st.cache_data(persist='disk')
def get_info_df(df):
    buf = io.StringIO()
    df.info(buf=buf)
    s = buf.getvalue()
    lines = [line.split() for line in s.splitlines()[3:-2]]
    return pd.DataFrame(lines)

@st.cache_data(persist='disk')
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

@st.cache_data(persist='disk')
def create_histogram(df, column):
    chart = alt.Chart(df).mark_bar().encode(
        alt.X(f"{column}:Q", bin=True),
        y='count()'
    ).properties(
        title=f'Histogram of {column}',
        width=300,
        height=300
    )
    return chart

@st.cache_resource
def create_histogram_grid(features_num, num_cols, df):
    rows = len(features_num) // num_cols + (len(features_num) % num_cols > 0)
    for i in range(rows):
        cols = st.columns(num_cols)
        for j in range(num_cols):
            idx = i * num_cols + j
            if idx < len(features_num):
                col_name = features_num[idx]
                with cols[j]:
                    st.altair_chart(create_histogram(df, col_name), use_container_width=True)

@st.cache_resource
def print_outliers(num_cols, df):
    # Check for outliers and handle them
    for col in num_cols:
        outliers = detect_outliers(df, col)
        st.text(f"Outliers detected in {col}: {len(outliers) if not outliers.empty else 0}")
        # Capping outliers
        cap = df[col].quantile(0.95)
        floor = df[col].quantile(0.05)
        df[col] = np.where(df[col] > cap, cap, df[col])
        df[col] = np.where(df[col] < floor, floor, df[col])

@st.cache_resource
def plot_correlation_matrix(data):
    import seaborn as sns
    try:
        correlation_matrix = data.corr()
        st.text(correlation_matrix)
    except:
        st.text('Error')

st.title('British Airways bookings analysis')
st.markdown('In this project we will analyse customer bookings data for Brith Airways , prepare it for training and finally train a model predict whether a customer completes a booking or not.')
df = pd.read_csv('data/british_airways_bookings.csv', encoding="ISO-8859-1", index_col=0, header=0)

st.dataframe(df.head())

st.markdown('First we try to understand what the dataset looks like')
st.dataframe(get_info_df(df), width=920)
st.markdown(
    '''
We can see that the dataset has a total of `13` columns, 50K rows , `5 integer`, `1 float` and 
`5 categorical` features. Next, lets see the data summary for further exploration.
'''
)
st.dataframe(df.describe())
st.markdown(
    '''
The table above gives us an overview of the data. We can see that all columns have 50K Non Null values which means
there is not missing data. The other metrics give us useful insights about each feature, 
e.g total number of `non-null` rows, `average value`, `standard deviation`, `minimum value`, `maximum value`, 
and the `percentiles` of the distribution. Next, we need to check how the data in each variable is distributed to see if any variables are skewed or not.
'''
)


# Streamlit app
st.title("Histograms of Numerical Features")

# Select numerical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).nunique().loc[lambda x: x > 2].index.tolist()

create_histogram_grid(features_num=numerical_columns, num_cols=3, df=df)

st.markdown(
    '''
From the histograms above we can see that the data is highly skewed. To fix the scaling we need to 
apply a transformation strategy. But before we apply a transformation, we need to check for outliers and 
remove them.
'''
)

continuous_features = df.select_dtypes(include=['int64', 'float64']).nunique().loc[lambda x: x > 2].index.tolist()
print_outliers(num_cols=continuous_features, df=df)


st.markdown(
    '''
A large number of outliers were detected so lets apply the Box-Cox transfrmation and feature scaling since 
it can work for right or left skewed dataset and requires positive values.
'''
)
df_transformed = df.copy()
continuous_features_transformed = df_transformed.select_dtypes(include=['int64', 'float64']).nunique().loc[lambda x: x > 2].index.tolist()
categorical_features = ['sales_channel', 'trip_type', 'flight_day', 'route', 'booking_origin']
for col in continuous_features_transformed:
    df_transformed[col] = np.log1p(df[col])


scaler = StandardScaler()
df_transformed[continuous_features_transformed] = scaler.fit_transform(df_transformed[continuous_features_transformed])
create_histogram_grid(features_num=continuous_features_transformed, num_cols=3, df=df_transformed)
st.markdown(
    '''
    After applying a transformation and scaling the features our data looks more normally distributed which
    means most of the outliers are gone. Now we need to apply one-hot-encoding to convert the categorical features.
'''
)
df_encoded = pd.get_dummies(df_transformed, columns = categorical_features,)
st.dataframe(df_encoded.head())

st.markdown(
    f'''
   Our data is now clean but before we proceed with training a model we first need to 
   check if any of the features are correlated with each other.
'''
)

plot_correlation_matrix(data=df_transformed[continuous_features_transformed])

st.markdown(
    f'''
   There doesn't seem to be any strong correlations among features. Now we check correlation between categorical variables using the chi2 contigency from scipy.
'''
)

def categorical_corr_matrix(df, cat_vars, alpha=0.05):
    """
    Compute the Chi-squared correlation matrix for categorical variables.

    Parameters:
    - df: pd.DataFrame containing the data
    - cat_vars: list of categorical variable names
    - alpha: significance level for the Chi-squared test

    Returns:
    - pd.DataFrame with p-values and significance indication
    """
    # Create an empty DataFrame to hold p-values
    corr_matrix = pd.DataFrame(index=cat_vars, columns=cat_vars)

    # Loop through each pair of categorical variables
    for i in range(len(cat_vars)):
        for j in range(len(cat_vars)):
            if i == j:
                corr_matrix.iloc[i, j] = 1.0  # Self-correlation
            else:
                # Create a contingency table
                contingency_table = pd.crosstab(df[cat_vars[i]], df[cat_vars[j]])
                # Perform the Chi-squared test
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                corr_matrix.iloc[i, j] = p

    # Indicate significance
    significance_matrix = corr_matrix.applymap(lambda x: 'associated' if x < alpha else 'not associated')

    return corr_matrix, significance_matrix


corr_matrix, significance_matrix = categorical_corr_matrix(df, categorical_features)

st.dataframe(corr_matrix)
st.dataframe(significance_matrix)

# if p < alpha:
#     st.text("Reject the null hypothesis - there is a significant association between the variables.")
# else:
#     st.text("Fail to reject the null hypothesis - no significant association between the variables.")