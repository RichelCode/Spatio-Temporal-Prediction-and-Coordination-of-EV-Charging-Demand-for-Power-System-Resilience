#!/usr/bin/env python
# coding: utf-8

# #  Spatio-Temporal Prediction and Coordination of EV Charging Demand for Power System Resilience

# ## Research Objectives
# 
# Recent studies have explored electric vehicles (EVs) from different perspectives, ranging from estimating vehicle range based on battery capacity, model specifications, and internal components (Ahmed et al., 2022) to forecasting charging behavior using machine learning methods such as Random Forest and SVM with factors like previous payment data, weather, and traffic (Shahriar et al., 2020). In parallel, research on smart cities has focused on managing traffic flow efficiently to reduce congestion and energy consumption (Dymora, Mazurek, & Jucha, 2024).
# 
# Building on these insights, this study links traffic dynamics with EV energy consumption to better predict when and where charging demand will arise. By integrating spatio-temporal traffic features with deep learning models, the goal is to anticipate EV charging needs in real time and enable coordinated charging strategies that support overall power system resilience.
# 

# ## Load Required Libraries 

# In[190]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from scipy import stats
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random


# ## Load and Clean the Data 

# In[81]:


df = pd.read_csv("cleaned_traffic_data.csv")


# ## How the data looks directly from PEMS

# In[82]:


df.head()


# ### We ignore and remove features that contain only NAN values, and maintain the other features.

# In[83]:


# Define the final selected columns
selected_columns = [
    "Timestamp", "Station", "Route", "Direction of Travel",
    "Total Flow", "Avg Speed", "% Observed","Samples","Lane Type"
]

# Keep only the selected columns
df = df[selected_columns]


# In[84]:


df


# ## Check the data types 

# In[85]:


df.dtypes


# ## Check the Percent of Missing Data in every feature 

# In[86]:


pd.set_option('display.float_format', '{:.4f}'.format)

missing_percent = (df.isna().sum() / len(df)) * 100
print(missing_percent)


# ## Imputation Strategy for Key Traffic Variables

# We decided to retain both the Average Speed and Total Flow features instead of dropping them because they are core variables that capture the essence of traffic dynamics. Average Speed reflects congestion levels and driving conditions, while Total Flow represents the number of vehicles passing a station—both directly influencing how traffic impacts EV range and, ultimately, charging demand. Dropping them would mean ignoring the very behaviors that determine how energy is consumed on the road. Even though these features had missing values, the patterns in traffic data are strongly structured in time and space, making them ideal candidates for informed imputation rather than removal.
# 
# For Average Speed, we applied a two-step temporal–spatial imputation strategy. First, we used forward and backward filling within each station to maintain continuity and preserve the natural hourly flow of traffic data. This approach works well because traffic speed rarely changes abruptly from one hour to the next unless influenced by an external event.
# 
# For Total Flow, the missingness was much lower, so a simpler approach was sufficient. We performed linear interpolation within each station to fill in small hourly gaps, ensuring that flow values remained smooth and representative of actual traffic movement. These imputation steps allowed us to preserve critical information about how vehicles move through the network without introducing artificial noise or bias. By reconstructing rather than discarding incomplete data, we maintained the integrity of the dataset and strengthened the foundation for accurate spatio-temporal modeling of EV charging demand and range prediction.

# In[87]:


df.sort_values(['Station', 'Timestamp'], inplace=True)
df['Avg Speed'] = df.groupby('Station')['Avg Speed'].ffill().bfill()


# In[88]:


df['Total Flow'] = df.groupby('Station')['Total Flow'].transform(
    lambda x: x.interpolate(method='linear')
)


# ## How the data Looks Like Now 

# In[89]:


df.head()


# # Feature Engineering 

# ## How do we incoporate and account for the spatial relationship in our data ?

# ### Incorporating Spatial Features into Linear Regression Models
# 
# To rigorously incorporate spatial features into linear regression models for the  PEMS-based traffic data, a systematic statistical approach is essential. Each record contains Station, Route, and Direction of Travel, which are sufficient for defining physical proximity. For each station, we partition the dataset by Route and Direction of Travel, then sort by Station ID. This ordering leverages the typical installation sequence of PEMS sensors and is supported in transportation literature when actual milepost data are unavailable.
# 
# For any station $s$ at time $t$, let the set of spatial neighbors (commonly the immediate upstream and downstream stations) be denoted as $\mathcal{N}(s)$. The flow at station $s$, $y_{s,t}$, is modeled as a function of its own temporal history and the flows of neighboring stations:
# 
# $$
# y_{s,t} = \beta_0 + \sum_{k=1}^{p} \beta_k x_{s,t-k} + \sum_{j \in \mathcal{N}(s)} \gamma_j y_{j,t-l_j} + \epsilon_{s,t}
# $$
# 
# Here:
# - $x_{s,t-k}$ represents temporal features (including lagged flows at station $s$)  
# - $y_{j,t-l_j}$ are flows at adjacent stations $j$, possibly with their own lags $l_j$   NB: for the neighboring stations we consider t, t-1 and t-2 
# - $\beta_k$ and $\gamma_j$ are regression coefficients  
# - $\epsilon_{s,t}$ is the error term  
# 
# This formulation captures spatial correlation as conditional dependence between adjacent sites, following spatial autoregressive principles within a linear regression framework.
# 
# In practice, using Python (pandas), after grouping by Route and Direction of Travel and sorting by Station, we generate for each observation the "upstream flow" and "downstream flow" variables, optionally at various lags (e.g., current or one-hour prior). These neighbor-based features are then included alongside traditional temporal predictors during model training. This ensures that spatial propagation and congestion effects, which are core to traffic dynamics, are represented in the model.
# 
# This approach ensures that even without explicit geo-coordinates, the regression model effectively captures spatial dependencies, leading to more accurate and interpretable traffic flow predictions across the studied transportation corridor.
# 

# In[90]:


import pandas as pd

grp_keys = ["Route", "Direction of Travel"]

#  Get unique stations per corridor with their spatial rank
corridor_stations = (
    df.groupby(grp_keys)["Station"]
    .unique()
    .apply(sorted)
    .reset_index()
    .rename(columns={"Station": "stations_list"})
)

# Explode to create a lookup table
neighbor_map = corridor_stations.explode("stations_list").reset_index(drop=True)
neighbor_map["station_rank"] = neighbor_map.groupby(grp_keys).cumcount()

# Create upstream/downstream mappings
neighbor_map["upstream_station"] = neighbor_map.groupby(grp_keys)["stations_list"].shift(1)
neighbor_map["downstream_station"] = neighbor_map.groupby(grp_keys)["stations_list"].shift(-1)

#  Merge back to original data
df = df.merge(
    neighbor_map.rename(columns={"stations_list": "Station"}),
    on=grp_keys + ["Station"],
    how="left"
)


# In[91]:


# See all unique combinations
df[['Route', 'Direction of Travel', 'Station', 'upstream_station', 'downstream_station']]\
  .drop_duplicates()\
  .sort_values(['Route', 'Direction of Travel', 'Station'])


# In[92]:


# Look at Station 311903 instead
df[df['Station'] == 311903].head()


# In[93]:


def merge_neighbor_flows(df, neighbor_col, new_col_prefix):
    '''
    Merge neighbor flows with t, t-1, t-2 lags
    
    Parameters:
    -----------
    df : DataFrame with upstream_station/downstream_station columns
    neighbor_col : str, name of the neighbor column ('upstream_station' or 'downstream_station')
    new_col_prefix : str, prefix for new columns ('upstream' or 'downstream')
    '''
    
    # Create lookup table
    neighbor_flow = df[['Station', 'Timestamp', 'Total Flow']].copy()
    neighbor_flow.rename(columns={'Station': neighbor_col}, inplace=True)
    
    # Merge current hour (t)
    df = df.merge(
        neighbor_flow.rename(columns={'Total Flow': f'{new_col_prefix}_flow'}),
        on=[neighbor_col, 'Timestamp'],
        how='left'
    )
    
    # Create lag 1 (t-1)
    df[f'{new_col_prefix}_flow_lag1'] = df.groupby(neighbor_col)[
        f'{new_col_prefix}_flow'
    ].shift(1)
    
    # Create lag 2 (t-2)
    df[f'{new_col_prefix}_flow_lag2'] = df.groupby(neighbor_col)[
        f'{new_col_prefix}_flow'
    ].shift(2)
    
    return df


# Now apply the function

# Apply to upstream neighbors
df = merge_neighbor_flows(df, 'upstream_station', 'upstream')

# Apply to downstream neighbors
df = merge_neighbor_flows(df, 'downstream_station', 'downstream')

# Verify the results
print("Spatial features created successfully!")
print(df[[
    'Timestamp', 'Station', 'Total Flow',
    'upstream_flow', 'upstream_flow_lag1', 'upstream_flow_lag2',
    'downstream_flow', 'downstream_flow_lag1', 'downstream_flow_lag2'
]].head(10))

# Check for missing values


# In[94]:


df = df.dropna(subset=[
'upstream_flow', 'upstream_flow_lag1', 'upstream_flow_lag2',
'downstream_flow', 'downstream_flow_lag1', 'downstream_flow_lag2'
])


# In[95]:


df.head()


# ## Including Temporal Features in our Data

# 
# ### 1. **Autoregressive Lags**
# - **Names:** flow_lag_1, flow_lag_2, flow_lag_3, flow_lag_6, flow_lag_12, flow_lag_24
# - **Role:** Capture short/intermediate/daily dependencies and persistence in traffic flow.
# - **Model inclusion:**
# $$ y_t = \beta_0 + \sum_{k \in \{1,2,3,6,12,24\}} \beta_k y_{t-k} + \epsilon_t $$
# 
# ### 2. **Rolling Statistics: Trend and Volatility**
# - **Names:** rolling_mean_24h, rolling_std_24h, rolling_min_24h, rolling_max_24h
# - **Role:** Quantify average, spread, and extremes over the last day to smooth volatility and capture local behavior.
# - **Formulas:**
#   - Mean: $$ \text{rolling\_mean\_24h}(t) = \frac{1}{24} \sum_{i=1}^{24} y_{t-i} $$
#   - Std Dev: $$ \text{rolling\_std\_24h}(t) = \sqrt{\frac{1}{24} \sum_{i=1}^{24}(y_{t-i} - \bar{y})^2} $$
# 
# ### 3. **Periodicity Features (Cyclic Encoding)**
# - **Names:** hour_sin, hour_cos, dow_sin, dow_cos, is_weekend, is_peak_hour
# - **Role:** Represent daily and weekly periodicities.
# - **Formulas:**
#   - Hour: $$ \text{hour\_sin}_t = \sin\left(\frac{2\pi h_t}{24}\right), \quad \text{hour\_cos}_t = \cos\left(\frac{2\pi h_t}{24}\right) $$
#   - DOW: $$ \text{dow\_sin}_t = \sin\left(\frac{2\pi d_t}{7}\right), \quad \text{dow\_cos}_t = \cos\left(\frac{2\pi d_t}{7}\right) $$
#   - Binary: is_weekend = 1 on weekends, is_peak_hour = 1 during commuting hours
# 
# ### 4. **Coefficient of Variation (CV_24h)**
# - **Name:** cv_24h
# - **Definition:** Ratio of standard deviation to mean over a 24-hour window:
#   - $$ \text{cv\_24h}(t) = \frac{\text{rolling\_std\_24h}(t)}{\text{rolling\_mean\_24h}(t)} $$
# - **Role:** Quantifies relative volatility; high CV signals instability in traffic flow. Used for diagnosing traffic state (stable, congested, fluctuating).
# 
# ## Mathematical and Applied Justification
# - **Autoregressive lags** capture natural persistence and delayed effects in traffic, standard in time-series analysis.
# - **Rolling statistics** (mean, std, min, max, CV) smooth local fluctuations and allow the model to react to recent volatility, supporting more robust predictions.
# - **Cyclic features** reflect the inherent periodicity in urban traffic, improving fit and interpretability, avoiding spurious jumps from one-hot hour/day encoding.
# - **Coefficient of variation** is widely used in transportation for characterizing the steadiness of flows and identifying transition states between free-flow and congestion.
# 
# 

# In[96]:


import numpy as np
import pandas as pd

# Ensure Timestamp column is datetime
if not np.issubdtype(df['Timestamp'].dtype, np.datetime64):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Autoregressive lags for each station
temp_lags = [1, 2, 3, 6, 12, 24]
for lag in temp_lags:
    df[f'flow_lag_{lag}'] = df.groupby('Station')['Total Flow'].shift(lag)

# Rolling statistics (24h), station-based
for stat, func in zip(['mean', 'std', 'min', 'max'], [np.mean, np.std, np.min, np.max]):
    df[f'rolling_{stat}_24h'] = (
        df.groupby('Station')['Total Flow']
          .transform(lambda x: x.rolling(24, min_periods=16).apply(func, raw=True))
    )

# Coefficient of Variation (CV = std/mean, 24h)
df['cv_24h'] = df['rolling_std_24h'] / (df['rolling_mean_24h'] + 1e-4)

# Cyclic time encodings
# Hour of day
hour = df['Timestamp'].dt.hour
# Day of week (Monday=0)
dow = df['Timestamp'].dt.dayofweek

df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

# Binary temporal flags
peak_hours = set([7,8,9,16,17,18])  # Common peak hour bands

df['is_peak_hour'] = hour.isin(peak_hours).astype(int)
df['is_weekend'] = df['Timestamp'].dt.dayofweek.isin([5,6]).astype(int)  # Saturday=5, Sunday=6


# In[97]:


df.head()


# In[98]:


required_features = [
    'flow_lag_1', 'flow_lag_2', 'flow_lag_3', 'flow_lag_6', 'flow_lag_12', 'flow_lag_24',
    'rolling_mean_24h', 'rolling_std_24h', 'rolling_min_24h', 'rolling_max_24h', 'cv_24h',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend', 'is_peak_hour',
    'Total Flow'
]
df = df.dropna(subset=required_features)


# In[99]:


df.head()


# # Exploratory Data Analysis

# ## Distribution of Total Flow

# In[100]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.histplot(df['Total Flow'].dropna(), bins=50, kde=True, color="#084594")
plt.title("Distribution of Hourly Total Flow", fontsize=16)
plt.xlabel("Hourly Total Flow", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.tight_layout()
plt.savefig("eda_total_flow_distribution.png", dpi=300)
plt.show()


# ## Temporal Autocorrelation (Hourly Lag Correlation Plot)

# In[101]:


from pandas.plotting import autocorrelation_plot
example_station = df['Station'].value_counts().index[0]
station_ts = df[df['Station']==example_station].sort_values('Timestamp')['Total Flow'].dropna()
plt.figure(figsize=(8,5))
autocorrelation_plot(station_ts)
plt.title(f"Autocorrelation of Hourly Flow — Station {example_station}", fontsize=16)
plt.xlabel("Lag (hour)", fontsize=14)
plt.tight_layout()
plt.savefig("eda_autocorrelation.png", dpi=300)
plt.show()


# ## Daily Cyclic Pattern (Boxplot by Hour of Day)

# In[102]:


import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(12,6))
hour = df['Timestamp'].dt.hour
sns.boxplot(x=hour, y=df['Total Flow'], palette="Blues")
plt.title("Hourly Traffic Flow Boxplot (Diurnal Cycle)", fontsize=16)
plt.xlabel("Hour of Day", fontsize=14)
plt.ylabel("Total Flow", fontsize=14)
plt.tight_layout()
plt.savefig("eda_hourly_boxplot.png", dpi=300)
plt.show()


# ## Spatial Feature Relationships (Scatter: Upstream vs. Own Flow)

# In[103]:


plt.figure(figsize=(8,6))
sns.scatterplot(x=df['upstream_flow'], y=df['Total Flow'], alpha=0.3, color="#2171b5")
plt.title("Upstream Flow vs. Own Station Flow", fontsize=16)
plt.xlabel("Upstream Station Flow", fontsize=14)
plt.ylabel("Current Station Flow", fontsize=14)
plt.tight_layout()
plt.savefig("eda_spatial_upstream_scatter.png", dpi=300)
plt.show()


# ## Average Traffic Flow by Hour Plot

# In[104]:


hourly_mean = df.groupby(df['Timestamp'].dt.hour)['Total Flow'].mean().reset_index()
plt.figure(figsize=(10,6))
sns.lineplot(x='Timestamp', y='Total Flow', data=hourly_mean, marker='o', color="#2171b5", linewidth=2)
plt.title('Average Hourly Traffic Flow Across All Stations', fontsize=16)
plt.xlabel('Hour of Day', fontsize=14)
plt.ylabel('Mean Flow (vehicles/hour)', fontsize=14)
plt.xticks(range(0,24))
plt.tight_layout()
plt.savefig('eda_average_traffic_flow_by_hour.png', dpi=300)
plt.show()


# ##  Weekly Cycle (Violin Plot of Flow by Weekday)

# In[105]:


plt.figure(figsize=(10,6))
dow = df['Timestamp'].dt.dayofweek
sns.violinplot(x=dow, y=df['Total Flow'], palette="Blues")
plt.title("Traffic Flow by Day of Week", fontsize=16)
plt.xlabel("Day of Week (Mon=0)", fontsize=14)
plt.ylabel("Total Flow", fontsize=14)
plt.tight_layout()
plt.savefig("eda_weekday_violinplot.png", dpi=300)
plt.show()


# ## Rolling Mean Trend for a Major Station

# In[106]:


major_station = df['Total Flow'].groupby(df['Station']).mean().idxmax()
station_data = df[df['Station']==major_station].sort_values('Timestamp')
plt.figure(figsize=(12,6))
plt.plot(station_data['Timestamp'], station_data['rolling_mean_24h'], color="#6baed6", linewidth=2)
plt.title(f"Rolling 24h Mean Traffic Flow — Station {major_station}", fontsize=16)
plt.xlabel("Datetime", fontsize=14)
plt.ylabel("Rolling Mean Flow", fontsize=14)
plt.tight_layout()
plt.savefig("eda_rolling_mean_24h.png", dpi=300)
plt.show()


# ## Median Coefficient of Variation (24h) by Station

# In[107]:


# Aggregate CV per station (median is robust to outliers)
cv_summary = df.groupby('Station')['cv_24h'].median().reset_index()
cv_summary = cv_summary.sort_values('cv_24h', ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(y='Station', x='cv_24h', data=cv_summary,
            palette=sns.color_palette('Blues', n_colors=len(cv_summary)))
plt.title('Median Coefficient of Variation (24h) by Station', fontsize=16)
plt.xlabel('Coefficient of Variation (CV, 24h) [unitless]', fontsize=14)
plt.ylabel('Station', fontsize=14)
plt.tight_layout()
plt.savefig('eda_cv24h_station_barplot.png', dpi=300)
plt.show()


# ## Volatility and Stability (CV_24h Distribution by Station)

# In[108]:


plt.figure(figsize=(10,6))
sns.boxplot(x=df['Station'], y=df['cv_24h'], palette="Blues")
plt.title("Coefficient of Variation (24h) Across Stations", fontsize=16)
plt.xlabel("Station", fontsize=14)
plt.ylabel("CV over 24h Window", fontsize=14)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("eda_station_cv24h_boxplot.png", dpi=300)
plt.show()


# # MODEL BUILDING 

# 
# ## Linear Regression for Traffic Flow Forecasting
# 
# Linear regression is a foundational supervised learning method that models the relationship between a target variable (future traffic flow) and one or more predictor features. It assumes this relationship to be linear, represented mathematically as:
# 
# $$
# y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_p x_p + \varepsilon
# $$
# 
# where (y) is the predicted traffic flow at a given future time, (x_1, \ldots, x_p) are engineered features (spatial, temporal, and raw operational predictors), (\beta_0) is the intercept, (\beta_1, \ldots, \beta_p) are regression coefficients, and (\varepsilon) is the error term.
# 
# ## Modeling Plan
# 
# 1. **Feature Set Exploration**
#    We will organize features into distinct groups: baseline (raw features), temporal (lags, cycles, rolling statistics), spatial (neighbor flows and lags), and their combinations. This enables rigorous comparison to determine which sets best drive predictive accuracy.
# 
# 2. **Multi-horizon Prediction**
#    The model will be trained to predict (y_{t+72}), the traffic flow 72 hours ahead. Using the same trained model, we will also forecast flow for 12, 24, and 48-hour horizons ((y_{t+12}, y_{t+24}, y_{t+48})) by inputting corresponding lagged and engineered features.
# 
# 3. **Mathematical Details**
#    For each horizon, the regression is formulated as:
# 
#    $$
#    y_{t+H} = \beta_0 + \sum_{j=1}^{p} \beta_j x_{j,t} + \varepsilon_{t+H}
#    $$
# 
#    where (H = 12, 24, 48, 72) and (x_{j,t}) are the relevant predictors available at time (t).
# 
# 4. **Model Evaluation**
# 
#    * We evaluate performance using standard metrics: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Coefficient of Determination ((R^2)) for each prediction horizon.
#    * Diagnostic scatter and time-series plots of predicted vs actual flows will visually and quantitatively assess fit.
# 
# ## Why This Approach?
# 
# * **Interpretability:** Linear regression provides direct mappings between feature changes and expected flow responses ((\beta_j)), valuable for understanding traffic drivers.
# * **Transparency:** Models with grouped features allow us to report effect sizes and statistical significance for publication.
# * **Multi-horizon generalizability:** Training and evaluating at varied forecast lengths tests practical utility and temporal robustness of engineered features.
# 
# 

# In[109]:


df.head()


# 
# ## Preprocessing Summary and Mathematical Rationale
# 
# ### 1. Chronological Train, Validation, and Test Split
# 
# The dataset is partitioned strictly by time to prevent look-ahead bias and preserve temporal causality:
# 
# * **Training:** Oct 1 – Nov 16
# * **Validation:** Nov 17 – Dec 11
# * **Test:** Dec 12 – Dec 31
# 
# This ensures that all model learning is based only on information available prior to the future periods being predicted.
# 
# 
# ## 2. Target Encoding for the Route Feature
# 
# To incorporate route-specific behavior while avoiding overfitting on rare categories, a smoothed target encoding is applied.
# For each route ( i ), the encoded value is:
# 
# $$
# \frac{
# n_i \cdot \bar{y}*{i,\text{train}}
# +
# \lambda \cdot \bar{y}*{\text{global,train}}
# }{
# n_i + \lambda
# }
# $$
# 
# 
# ### 3. One-Hot Encoding of Low-Cardinality Features
# 
# Categorical features with low cardinality (such as **Lane Type** or **Direction of Travel**) are converted into indicator variables using one-hot encoding with `drop_first=True`. This prevents unnecessary multicollinearity and keeps feature dimensionality minimal.
# 
# 
# ### **4. Feature Alignment and Cleaning**
# 
# After all encoding steps:
# 
# * Each split (train, validation, test) contains **identical feature columns**
# * Unseen categories in validation or test are assigned zeros
# * Non-feature fields such as timestamps, IDs, or text descriptors are removed
# 
# This ensures that the model receives a consistent input structure across all periods.
# 
# 
# ### **5. Standardization of Continuous Variables**
# 
# Continuous variables are standardized using statistics from the training set only:
# 
# $$
# \text{standardized}(x) =
# \frac{
# x - \mu_{\text{train}}
# }{
# \sigma_{\text{train}}
# }
# $$
# 
# Applying training-only statistics prevents information leakage from future periods.
# 
# 
# ### **6. Prevention of Data Leakage**
# 
# All preprocessing components including target encoding, one-hot encoding, and scaling—are **fit on the training set exclusively**.
# The fitted transformers are then applied to the validation and test sets **without recomputing** statistics.
# 
# This approach ensures that no information from future time periods influences the training process.

# In[177]:


total_rows = len(df)
train_size = int(0.70 * total_rows)
val_size = int(0.20 * total_rows)
train_df = df.iloc[:train_size].copy()
val_df = df.iloc[train_size:train_size+val_size].copy()
test_df = df.iloc[train_size+val_size:].copy()


# In[180]:


# Calculate route means from TRAINING data ONLY
route_means = train_df.groupby('Route')['Total Flow'].agg(['mean', 'count']).reset_index()
route_means.columns = ['Route', 'mean_flow', 'count']

global_mean = train_df['Total Flow'].mean()
lambda_smooth = 1.0

route_means['Route_Encoded'] = (
    route_means['count'] * route_means['mean_flow'] + lambda_smooth * global_mean
) / (route_means['count'] + lambda_smooth)

route_encoding_map = dict(zip(route_means['Route'], route_means['Route_Encoded']))

# Apply to all sets
for df_set_name, df_set in [('train_df', train_df), ('val_df', val_df), ('test_df', test_df)]:
    df_set['Route_Encoded'] = df_set['Route'].map(route_encoding_map)
    df_set['Route_Encoded'].fillna(global_mean, inplace=True)
    df_set.drop('Route', axis=1, inplace=True)
    print(f" Route encoded for {df_set_name}")


# In[181]:


categorical_cols = ['Lane Type', 'Direction of Travel']
train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True,
prefix=['Lane', 'Direction'])
val_df = pd.get_dummies(val_df, columns=categorical_cols, drop_first=True,
prefix=['Lane', 'Direction'])
test_df = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True,
prefix=['Lane', 'Direction'])
# Align all sets to have same columns
train_df, val_df = train_df.align(val_df, join='left', axis=1, fill_value=0)
train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)


# In[194]:


# Recombine all splits into FULL dataset
df_full = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Sort by timestamp
df_full = df_full.sort_values('Timestamp').reset_index(drop=True)

print(f"✓ Full dataset: {len(df_full):,} rows")

# Create multi-horizon targets
HORIZONS = [12, 24, 48, 72]

for h in HORIZONS:
    df_full[f'Target_t_plus_{h}'] = df_full['Total Flow'].shift(-h)

# Drop rows where ANY target is NaN (ensures alignment)
target_columns = [f'Target_t_plus_{h}' for h in HORIZONS]
df_full = df_full.dropna(subset=target_columns).reset_index(drop=True)

print(f"✓ Multi-horizon targets created")
print(f"  Targets: {target_columns}")
print(f"  Rows remaining: {len(df_full):,}")


# In[195]:


# SEPARATE FEATURES AND TARGETS

# Drop columns that are NOT features
cols_to_drop = ['Total Flow', 'Timestamp', 'Station'] + target_columns
cols_to_drop = [c for c in cols_to_drop if c in df_full.columns]

X_full = df_full.drop(cols_to_drop, axis=1, errors='ignore')

print(f"✓ Features extracted: {X_full.shape}")
print(f"  Features: {list(X_full.columns[:10])}...")

# Create TARGET DICTIONARIES (this is critical!)
y_full_dict = {}
for h in HORIZONS:
    y_full_dict[h] = df_full[f'Target_t_plus_{h}'].copy()

print(f"✓ Target dictionaries created")
for h in HORIZONS:
    print(f"  y_full_dict[{h}]: {y_full_dict[h].shape}")


# In[196]:


# STANDARDIZE FEATURES

# Identify numeric columns
all_numeric = X_full.select_dtypes(include=[np.number]).columns.tolist()
binary_features = [col for col in all_numeric if X_full[col].nunique() <= 2]
continuous_features = [col for col in all_numeric if col not in binary_features]

print(f"✓ Feature types identified:")
print(f"  Total numeric: {len(all_numeric)}")
print(f"  Binary: {len(binary_features)}")
print(f"  Continuous: {len(continuous_features)}")

# Calculate 70% cutoff
train_cutoff = int(0.70 * len(X_full))
print(f"✓ Scaler fit on first 70%: {train_cutoff:,} rows")

# Fit scaler ONLY on first 70%
scaler = StandardScaler()
scaler.fit(X_full.iloc[:train_cutoff][continuous_features])

# Apply to ALL data
X_full_scaled = X_full.copy()
X_full_scaled[continuous_features] = scaler.transform(X_full[continuous_features])

print(f"✓ Standardization complete")
print(f"  X_full_scaled: {X_full_scaled.shape}")


# In[197]:


# GET FINAL FEATURE LIST

effective_features = list(X_full_scaled.columns)

print(f"✓ Effective features: {len(effective_features)}")
print(f"  Shape: {X_full_scaled.shape}")

# Verify no NaN
nan_count = X_full_scaled.isnull().sum().sum()
if nan_count > 0:
    raise ValueError(f"ERROR: {nan_count} NaN values found!")
print(f"✓ No NaN values")


# ## EXPANDING WINDOW CROSS_VALIDATION FOR TIME SERIES DATA

# In[198]:


# CREATE EXPANDING WINDOW CV SPLITS

# This function creates 5 expanding window CV folds
def create_expanding_window_cv_splits(n_total, n_splits=5, test_fraction=0.10, val_fraction=0.10):
    """
    Create expanding window cross-validation splits.
    For time series: Train window grows, validation and test stay fixed size
    """
    splits = []
    test_n = int(n_total * test_fraction)
    val_n = int(n_total * val_fraction)
    step = int((n_total - test_n - val_n) / n_splits)
    
    for i in range(n_splits):
        train_end = (i + 1) * step
        val_start = train_end
        val_end = val_start + val_n
        test_start = val_end
        
        if test_start + test_n > n_total:
            break
        
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        test_idx = np.arange(test_start, test_start + test_n)
        
        splits.append((train_idx, val_idx, test_idx))
    
    return splits

# Create splits
cv_splits = create_expanding_window_cv_splits(
    n_total=len(X_full_scaled),
    n_splits=5,
    test_fraction=0.10,
    val_fraction=0.10
)

print(f"✓ Created {len(cv_splits)} expanding window CV folds:")
for i, (tr, val, te) in enumerate(cv_splits):
    print(f"  Fold {i+1}: Train {len(tr):,} | Val {len(val):,} | Test {len(te):,}")

# Verify indices are correct
max_idx = max([max(te) for _, _, te in cv_splits])
print(f"\n✓ Index verification:")
print(f"  Max index in CV: {max_idx:,}")
print(f"  Dataset length: {len(X_full_scaled):,}")
print(f"  Valid: {max_idx < len(X_full_scaled)}")


# # Linear Regression (Elastic Net)

# ## Direct Strategy - Separate Models per Horizon

# In[200]:


# ========================================
# DIRECT (Separate Models)
# ========================================
class DirectStrategy:
    """
    Train separate linear regression model for each forecast horizon.
    Each model is optimized independently with Elastic Net regularization.
    """
    def __init__(self, horizons, features, alpha=0.5, lambda_=0.1):
        self.horizons = horizons
        self.features = features
        self.alpha = alpha
        self.lambda_ = lambda_
        self.models = {}
        self.cv_results = {h: [] for h in horizons}

    def fit_and_evaluate(self, X_scaled, y_dict, cv_splits, scaler_stats):
        """
        Train and evaluate models for each horizon using expanding window CV.
        """
        results_list = []

        for horizon in self.horizons:
            print(f"\n{'='*80}")
            print(f"DIRECT STRATEGY: Training horizon t+{horizon}")
            print(f"{'='*80}")

            models_cv = []
            fold_results = []

            for fold, (train_idx, val_idx, test_idx) in enumerate(cv_splits):
                print(f"\nFold {fold+1}/{len(cv_splits)}:")

                # Prepare data for this fold
                X_train_fold = X_scaled.iloc[train_idx][self.features].copy()
                y_train_fold = y_dict[horizon].iloc[train_idx].copy()

                X_val_fold = X_scaled.iloc[val_idx][self.features].copy()
                y_val_fold = y_dict[horizon].iloc[val_idx].copy()

                X_test_fold = X_scaled.iloc[test_idx][self.features].copy()
                y_test_fold = y_dict[horizon].iloc[test_idx].copy()

                # Train model with Elastic Net regularization
                model = ElasticNet(
                    alpha=self.lambda_,
                    l1_ratio=self.alpha,
                    max_iter=10000,
                    random_state=42
                )
                model.fit(X_train_fold, y_train_fold)
                models_cv.append(model)

                # Predictions
                y_pred_val = model.predict(X_val_fold)
                y_pred_test = model.predict(X_test_fold)

                # Metrics
                rmse_val = np.sqrt(mean_squared_error(y_val_fold, y_pred_val))
                mae_val = mean_absolute_error(y_val_fold, y_pred_val)
                mape_val = mean_absolute_percentage_error(y_val_fold, y_pred_val)
                r2_val = r2_score(y_val_fold, y_pred_val)

                rmse_test = np.sqrt(mean_squared_error(y_test_fold, y_pred_test))
                mae_test = mean_absolute_error(y_test_fold, y_pred_test)
                mape_test = mean_absolute_percentage_error(y_test_fold, y_pred_test)
                r2_test = r2_score(y_test_fold, y_pred_test)

                fold_results.append({
                    'fold': fold + 1,
                    'horizon': horizon,
                    'strategy': 'Direct',
                    'val_rmse': rmse_val,
                    'val_mae': mae_val,
                    'val_mape': mape_val,
                    'val_r2': r2_val,
                    'test_rmse': rmse_test,
                    'test_mae': mae_test,
                    'test_mape': mape_test,
                    'test_r2': r2_test,
                    'n_features': len(self.features),
                    'model': model,
                    'y_pred_val': y_pred_val,
                    'y_pred_test': y_pred_test
                })

                print(f" Val RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}, R2: {r2_val:.4f}")
                print(f" Test RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}, R2: {r2_test:.4f}")

            self.models[horizon] = models_cv
            self.cv_results[horizon] = fold_results
            results_list.extend(fold_results)

        return pd.DataFrame(results_list)


direct_strategy = DirectStrategy(
    horizons=HORIZONS,
    features=effective_features,
    alpha=0.5,
    lambda_=0.1
)

print("Training Direct Strategy...")

direct_results = direct_strategy.fit_and_evaluate(
    X_scaled=X_full_scaled,      # ← CRITICAL: Use FULL dataset
    y_dict=y_full_dict,          # ← CRITICAL: Use ALL horizons
    cv_splits=cv_splits,         # ← 5 CV folds
    scaler_stats=None 
)

print("\n✓ Direct Strategy complete")
direct_summary = direct_results.groupby('horizon')[['test_rmse', 'test_mae', 'test_r2']].mean()
print(direct_summary)


# ## MIMO Strategy - Single Model, Multiple Outputs

# In[207]:


# ========================================
#  MIMO (Multiple-Input Multiple-Output)
# ========================================
class MIMOStrategy:
    """
    Single linear regression model that predicts all horizons simultaneously.
    Captures inter-horizon dependencies through shared feature space.
    """
    def __init__(self, horizons, features, alpha=0.5, lambda_=0.1):
        self.horizons = horizons
        self.features = features
        self.alpha = alpha
        self.lambda_ = lambda_
        self.models = {}
        self.cv_results = []

    def fit_and_evaluate(self, X_scaled, y_dict, cv_splits):
        """
        Train MIMO model using expanding window CV.
        """
        results_list = []

        print(f"\n{'='*80}")
        print(f"MIMO STRATEGY: Single model predicting t+{self.horizons}")
        print(f"{'='*80}")

        for fold, (train_idx, val_idx, test_idx) in enumerate(cv_splits):
            print(f"\nFold {fold+1}/{len(cv_splits)}:")

            # Prepare data for this fold
            X_train_fold = X_scaled.iloc[train_idx][self.features].copy()
            X_val_fold = X_scaled.iloc[val_idx][self.features].copy()
            X_test_fold = X_scaled.iloc[test_idx][self.features].copy()

            # Create multi-target array
            y_train_multi = np.column_stack([
                y_dict[h].iloc[train_idx].values for h in self.horizons
            ])
            y_val_multi = np.column_stack([
                y_dict[h].iloc[val_idx].values for h in self.horizons
            ])
            y_test_multi = np.column_stack([
                y_dict[h].iloc[test_idx].values for h in self.horizons
            ])

            # Train single multi-output model
            model = ElasticNet(
                alpha=self.lambda_,
                l1_ratio=self.alpha,
                max_iter=10000,
                random_state=42
            )

            # Use MultiTaskElasticNet for true multi-output ElasticNet
            from sklearn.linear_model import MultiTaskElasticNet
            model = MultiTaskElasticNet(
                alpha=self.lambda_,
                l1_ratio=self.alpha,
                max_iter=10000,
                random_state=42
            )
            model.fit(X_train_fold, y_train_multi)

            # Predictions
            y_pred_val = model.predict(X_val_fold)
            y_pred_test = model.predict(X_test_fold)

            # Calculate metrics for each horizon
            for h_idx, h in enumerate(self.horizons):
                y_val_h = y_val_multi[:, h_idx]
                y_test_h = y_test_multi[:, h_idx]
                y_pred_val_h = y_pred_val[:, h_idx]
                y_pred_test_h = y_pred_test[:, h_idx]

                rmse_val = np.sqrt(mean_squared_error(y_val_h, y_pred_val_h))
                mae_val = mean_absolute_error(y_val_h, y_pred_val_h)
                mape_val = mean_absolute_percentage_error(y_val_h, y_pred_val_h)
                r2_val = r2_score(y_val_h, y_pred_val_h)

                rmse_test = np.sqrt(mean_squared_error(y_test_h, y_pred_test_h))
                mae_test = mean_absolute_error(y_test_h, y_pred_test_h)
                mape_test = mean_absolute_percentage_error(y_test_h, y_pred_test_h)
                r2_test = r2_score(y_test_h, y_pred_test_h)
                
                print(f"  Horizon t+{h}: Val RMSE={rmse_val:.4f}, MAE={mae_val:.4f}, R2={r2_val:.4f} | Test RMSE={rmse_test:.4f}, MAE={mae_test:.4f}, R2={r2_test:.4f}")

                results_list.append({
                    'fold': fold + 1,
                    'horizon': h,
                    'strategy': 'MIMO',
                    'val_rmse': rmse_val,
                    'val_mae': mae_val,
                    'val_mape': mape_val,
                    'val_r2': r2_val,
                    'test_rmse': rmse_test,
                    'test_mae': mae_test,
                    'test_mape': mape_test,
                    'test_r2': r2_test,
                    'n_features': len(self.features),
                    'model': model if h_idx == 0 else None
                })

            self.models[fold] = model

        self.cv_results = results_list
        return pd.DataFrame(results_list)


# TRAIN MIMO STRATEGY

# Your MIMOStrategy class should be defined already in your notebook

mimo_strategy = MIMOStrategy(
    horizons=HORIZONS,
    features=effective_features,
    alpha=0.5,
    lambda_=0.1
)

print("Training MIMO Strategy...")

mimo_results = mimo_strategy.fit_and_evaluate(
    X_scaled=X_full_scaled,      # ← CRITICAL: Use FULL dataset
    y_dict=y_full_dict,          # ← CRITICAL: Use ALL horizons
    cv_splits=cv_splits  # ← Same 5 folds
   
)

print("\n✓ MIMO Strategy complete")
mimo_summary = mimo_results.groupby('horizon')[['test_rmse', 'test_mae', 'test_r2']].mean()
print(mimo_summary)



# ## COMPARE STRATEGIES

# In[208]:


#  COMPARE STRATEGIES

# Combine all results
all_results = pd.concat([
    direct_results.assign(strategy='Direct'),
    mimo_results.assign(strategy='MIMO')
], ignore_index=True)

# Summary comparison
print("\n" + "="*100)
print("STRATEGY COMPARISON")
print("="*100)

comparison = all_results.groupby(['strategy', 'horizon']).agg({
    'test_rmse': ['mean', 'std'],
    'test_mae': ['mean', 'std'],
    'test_r2': ['mean', 'std']
}).round(4)

print(comparison)

# Save results
all_results.to_csv('multi_horizon_final_results.csv', index=False)
print("\n✓ Results saved to: multi_horizon_final_results.csv")



# These two strategies had very similar results for the linear regression.

# Model performance was evaluated using 5-fold expanding window cross-validation, with mean and standard deviation of test MAE and RMSE calculated over each fold for every forecast horizon, in line with recent benchmarking standards for time series forecasting.

# ## Visualizations

# In[215]:


# Data
horizons = np.array([12, 24, 48, 72])
mae_means = np.array([834.1925, 834.1196, 834.3554, 834.3956])
mae_stds  = np.array([21.3097, 21.3167, 21.3017, 21.3644])
rmse_means = np.array([1164.0771, 1164.1017, 1164.2831, 1164.4685])
rmse_stds  = np.array([37.1661, 37.2029, 37.1840, 37.1918])

plt.rcParams.update({
    'axes.labelsize': 9,       # Journal standard font size
    'font.size': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.titlesize': 10,
    'axes.linewidth': 0.8,
    'figure.figsize': [6, 3.2], # Side-by-side, compact
    'savefig.dpi': 300

})

fig, axes = plt.subplots(1, 2, constrained_layout=True)

# RMSE Panel
axes[0].errorbar(horizons, rmse_means, yerr=rmse_stds, fmt='o', color='#25446b',
    capsize=4, elinewidth=1, markeredgewidth=1, markersize=6)
axes[0].set_xlabel('Forecast Horizon (hours)', fontweight='bold')
axes[0].set_ylabel('RMSE (vehicles/hour)', fontweight='bold')
axes[0].set_title('(a) RMSE by Horizon', loc='left', fontweight='bold')
axes[0].grid(True, linestyle=':', alpha=0.15)
axes[0].set_xlim(10, 74)
axes[0].set_ylim(1130, 1190)
for x, y in zip(horizons, rmse_means):
    axes[0].annotate(f'{y:.1f}', (x, y), xytext=(0,7), textcoords="offset points", ha='center', fontsize=8)

# MAE Panel
axes[1].errorbar(horizons, mae_means, yerr=mae_stds, fmt='o', color='#b0315c',
    capsize=4, elinewidth=1, markeredgewidth=1, markersize=6)
axes[1].set_xlabel('Forecast Horizon (hours)', fontweight='bold')
axes[1].set_ylabel('MAE (vehicles/hour)', fontweight='bold')
axes[1].set_title('(b) MAE by Horizon', loc='left', fontweight='bold')
axes[1].grid(True, linestyle=':', alpha=0.15)
axes[1].set_xlim(10, 74)
axes[1].set_ylim(800, 860)
for x, y in zip(horizons, mae_means):
    axes[1].annotate(f'{y:.1f}', (x, y), xytext=(0,7), textcoords="offset points", ha='center', fontsize=8)

# Export: vector PDF for journal submission (or PNG for review)
plt.savefig('Figure_MAE_RMSE_Horizons.pdf', bbox_inches='tight')
plt.savefig('Figure_MAE_RMSE_Horizons.png', bbox_inches='tight')
plt.show()


# Figure displays the mean and standard deviation of RMSE and MAE calculated over five expanding window cross-validation folds for each forecast horizon, reporting the robustness and accuracy of the Elastic Net baseline.
# Error bars denote standard deviation across folds.

# NB: We report the mean ± standard deviation of test set MAE (and RMSE) across five expanding window CV folds
# 
# Across 5 expanding window folds, our model's RMSE remained close to 1164 vehicles/hour, with a standard deviation of 37, indicating stable forecast accuracy and robustness to temporal variation.

# # Random Forest

# Random Forest is an ensemble learning method that captures complex non-linear interactions, spatial dependencies, and threshold effects inherent in traffic dynamics such as congestion onset, rush hour transitions, and weekend pattern shifts.

# ## Direct Strategy - Separate Models per Horizon (Random Forest)

# In[218]:


# ========================================
# RANDOM FOREST - DIRECT STRATEGY
# ========================================

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class RandomForestDirectStrategy:
    """
    Direct strategy: Train separate Random Forest model for each forecast horizon.
    Each model optimizes independently for its specific prediction window.
    
    Advantages over Linear Regression:
    - Captures non-linear relationships (congestion thresholds, rush hour effects)
    - Automatic interaction detection (no manual feature engineering needed)
    - Robust to outliers (split-based rather than MSE-based optimization)
    - No assumptions about functional form
    
    Parameters
    ----------
    horizons : list
        Forecast horizons in hours (e.g., [12, 24, 48, 72])
    features : list
        Feature column names to use for prediction
    n_estimators : int, default=200
        Number of trees in the forest
    max_features : str or float, default='sqrt'
        Number of features to consider at each split
        - 'sqrt': sqrt(n_features) ≈ 6 for 35 features
        - 'log2': log2(n_features) ≈ 5 for 35 features
        - float: fraction of features (e.g., 0.3 = 10-11 features)
    max_depth : int or None, default=None
        Maximum depth of trees (None = unlimited)
    min_samples_split : int, default=2
        Minimum samples required to split a node
    min_samples_leaf : int, default=1
        Minimum samples required at a leaf node
    random_state : int, default=42
        Random seed for reproducibility
    n_jobs : int, default=-1
        Number of CPU cores to use (-1 = all cores)
    oob_score : bool, default=True
        Whether to use out-of-bag samples to estimate R² on unseen data
    """
    
    def __init__(self, horizons, features, 
                 n_estimators=200, 
                 max_features='sqrt',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 random_state=42,
                 n_jobs=-1,
                 oob_score=True):
        
        self.horizons = horizons
        self.features = features
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.oob_score = oob_score
        
        # Storage for trained models and results
        self.models = {}  # {horizon: [model_fold1, model_fold2, ...]}
        self.cv_results = {h: [] for h in horizons}
        self.feature_importance = {}  # {horizon: importance_df}
        
    def fit_and_evaluate(self, X_scaled, y_dict, cv_splits):
        """
        Train and evaluate Random Forest models for each horizon using expanding window CV.
        
        Parameters
        ----------
        X_scaled : DataFrame
            Feature matrix (standardized or not - doesn't matter for RF)
        y_dict : dict
            Dictionary mapping horizons to target arrays
        cv_splits : list
            List of (train_idx, val_idx, test_idx) tuples for cross-validation
        
        Returns
        -------
        results_df : DataFrame
            Detailed results for all folds and horizons
        """
        
        results_list = []
        
        for horizon in self.horizons:
            print(f"\n{'='*80}")
            print(f"RANDOM FOREST DIRECT: Training horizon t+{horizon}")
            print(f"{'='*80}")
            
            models_cv = []
            importance_list = []
            
            for fold, (train_idx, val_idx, test_idx) in enumerate(cv_splits):
                print(f"\nFold {fold+1}/{len(cv_splits)}:")
                
                # Prepare data for this fold
                X_train_fold = X_scaled.iloc[train_idx][self.features].copy()
                y_train_fold = y_dict[horizon].iloc[train_idx].copy()
                
                X_val_fold = X_scaled.iloc[val_idx][self.features].copy()
                y_val_fold = y_dict[horizon].iloc[val_idx].copy()
                
                X_test_fold = X_scaled.iloc[test_idx][self.features].copy()
                y_test_fold = y_dict[horizon].iloc[test_idx].copy()
                
                # Train Random Forest model
                model = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_features=self.max_features,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state + fold,  # Different seed per fold
                    n_jobs=self.n_jobs,
                    oob_score=self.oob_score,
                    verbose=0
                )
                
                print(f"  Training RF with {self.n_estimators} trees, max_features={self.max_features}...")
                model.fit(X_train_fold, y_train_fold)
                models_cv.append(model)
                
                # Store feature importance for this fold
                importance_df = pd.DataFrame({
                    'feature': self.features,
                    'importance': model.feature_importances_,
                    'fold': fold + 1,
                    'horizon': horizon
                }).sort_values('importance', ascending=False)
                importance_list.append(importance_df)
                
                # Predictions
                y_pred_train = model.predict(X_train_fold)
                y_pred_val = model.predict(X_val_fold)
                y_pred_test = model.predict(X_test_fold)
                
                # Calculate metrics
                # Training metrics
                rmse_train = np.sqrt(mean_squared_error(y_train_fold, y_pred_train))
                mae_train = mean_absolute_error(y_train_fold, y_pred_train)
                r2_train = r2_score(y_train_fold, y_pred_train)
                
                # Validation metrics
                rmse_val = np.sqrt(mean_squared_error(y_val_fold, y_pred_val))
                mae_val = mean_absolute_error(y_val_fold, y_pred_val)
                mape_val = mean_absolute_percentage_error(y_val_fold, y_pred_val)
                r2_val = r2_score(y_val_fold, y_pred_val)
                
                # Test metrics
                rmse_test = np.sqrt(mean_squared_error(y_test_fold, y_pred_test))
                mae_test = mean_absolute_error(y_test_fold, y_pred_test)
                mape_test = mean_absolute_percentage_error(y_test_fold, y_pred_test)
                r2_test = r2_score(y_test_fold, y_pred_test)
                
                # OOB score (if enabled)
                oob_r2 = model.oob_score_ if self.oob_score else np.nan
                
                # Store results
                fold_result = {
                    'fold': fold + 1,
                    'horizon': horizon,
                    'strategy': 'RF_Direct',
                    'n_estimators': self.n_estimators,
                    'max_features': self.max_features,
                    'train_rmse': rmse_train,
                    'train_mae': mae_train,
                    'train_r2': r2_train,
                    'val_rmse': rmse_val,
                    'val_mae': mae_val,
                    'val_mape': mape_val,
                    'val_r2': r2_val,
                    'test_rmse': rmse_test,
                    'test_mae': mae_test,
                    'test_mape': mape_test,
                    'test_r2': r2_test,
                    'oob_r2': oob_r2,
                    'n_features': len(self.features),
                    'model': model,
                    'y_pred_val': y_pred_val,
                    'y_pred_test': y_pred_test,
                    'y_true_val': y_val_fold,
                    'y_true_test': y_test_fold
                }
                
                results_list.append(fold_result)
                
                # Print fold results
                print(f"  Train: RMSE={rmse_train:.2f}, MAE={mae_train:.2f}, R²={r2_train:.4f}")
                print(f"  Val:   RMSE={rmse_val:.2f}, MAE={mae_val:.2f}, R²={r2_val:.4f}")
                print(f"  Test:  RMSE={rmse_test:.2f}, MAE={mae_test:.2f}, R²={r2_test:.4f}")
                if self.oob_score:
                    print(f"  OOB R²: {oob_r2:.4f}")
            
            # Store models and feature importance for this horizon
            self.models[horizon] = models_cv
            self.cv_results[horizon] = [r for r in results_list if r['horizon'] == horizon]
            
            # Aggregate feature importance across folds
            importance_combined = pd.concat(importance_list)
            importance_summary = importance_combined.groupby('feature')['importance'].agg(['mean', 'std']).sort_values('mean', ascending=False)
            self.feature_importance[horizon] = importance_summary
            
            print(f"\n  Top 10 Most Important Features for t+{horizon}:")
            print(importance_summary.head(10).to_string())
        
        return pd.DataFrame(results_list)
    
    def get_feature_importance(self, horizon=None, top_n=10):
        """
        Get aggregated feature importance across CV folds.
        
        Parameters
        ----------
        horizon : int or None
            Specific horizon (if None, returns all horizons)
        top_n : int
            Number of top features to return
        
        Returns
        -------
        DataFrame or dict of DataFrames
        """
        if horizon is not None:
            return self.feature_importance[horizon].head(top_n)
        else:
            return {h: self.feature_importance[h].head(top_n) for h in self.horizons}


# ## Random Forest MIMO Strategy

# In[219]:


# ========================================
# RANDOM FOREST - MIMO STRATEGY
# ========================================

from sklearn.multioutput import MultiOutputRegressor

class RandomForestMIMOStrategy:
    """
    MIMO strategy: Single Random Forest that predicts all horizons simultaneously.
    Uses MultiOutputRegressor which trains separate trees for each output but
    shares bootstrap samples and feature subsets.
    
    Advantages:
    - Captures inter-horizon dependencies
    - Single model = faster training and smaller storage
    - Consistent predictions across horizons
    
    Disadvantages vs Direct:
    - Less flexibility to optimize per horizon
    - May underperform Direct at specific horizons
    
    Parameters are identical to RandomForestDirectStrategy.
    """
    
    def __init__(self, horizons, features,
                 n_estimators=200,
                 max_features='sqrt',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 random_state=42,
                 n_jobs=-1,
                 oob_score=True):
        
        self.horizons = horizons
        self.features = features
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.oob_score = oob_score
        
        self.models = {}
        self.cv_results = []
        self.feature_importance = {}
        
    def fit_and_evaluate(self, X_scaled, y_dict, cv_splits):
        """
        Train and evaluate MIMO Random Forest using expanding window CV.
        """
        
        results_list = []
        
        print(f"\n{'='*80}")
        print(f"RANDOM FOREST MIMO: Single model predicting t+{self.horizons}")
        print(f"{'='*80}")
        
        for fold, (train_idx, val_idx, test_idx) in enumerate(cv_splits):
            print(f"\nFold {fold+1}/{len(cv_splits)}:")
            
            # Prepare data
            X_train_fold = X_scaled.iloc[train_idx][self.features].copy()
            X_val_fold = X_scaled.iloc[val_idx][self.features].copy()
            X_test_fold = X_scaled.iloc[test_idx][self.features].copy()
            
            # Create multi-target arrays
            y_train_multi = np.column_stack([
                y_dict[h].iloc[train_idx].values for h in self.horizons
            ])
            y_val_multi = np.column_stack([
                y_dict[h].iloc[val_idx].values for h in self.horizons
            ])
            y_test_multi = np.column_stack([
                y_dict[h].iloc[test_idx].values for h in self.horizons
            ])
            
            # Create base Random Forest
            base_rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + fold,
                n_jobs=self.n_jobs,
                oob_score=self.oob_score,
                verbose=0
            )
            
            # Wrap with MultiOutputRegressor
            model = MultiOutputRegressor(base_rf, n_jobs=1)  # n_jobs=1 because base_rf already parallelizes
            
            print(f"  Training MIMO RF with {self.n_estimators} trees per output...")
            model.fit(X_train_fold, y_train_multi)
            self.models[fold] = model
            
            # Extract feature importance (average across outputs)
            importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
            importance_df = pd.DataFrame({
                'feature': self.features,
                'importance': importances,
                'fold': fold + 1
            }).sort_values('importance', ascending=False)
            
            if fold == 0:
                self.feature_importance['MIMO'] = importance_df
            
            # Predictions
            y_pred_train = model.predict(X_train_fold)
            y_pred_val = model.predict(X_val_fold)
            y_pred_test = model.predict(X_test_fold)
            
            # Calculate metrics for each horizon
            for h_idx, h in enumerate(self.horizons):
                y_train_h = y_train_multi[:, h_idx]
                y_val_h = y_val_multi[:, h_idx]
                y_test_h = y_test_multi[:, h_idx]
                
                y_pred_train_h = y_pred_train[:, h_idx]
                y_pred_val_h = y_pred_val[:, h_idx]
                y_pred_test_h = y_pred_test[:, h_idx]
                
                # Training metrics
                rmse_train = np.sqrt(mean_squared_error(y_train_h, y_pred_train_h))
                mae_train = mean_absolute_error(y_train_h, y_pred_train_h)
                r2_train = r2_score(y_train_h, y_pred_train_h)
                
                # Validation metrics
                rmse_val = np.sqrt(mean_squared_error(y_val_h, y_pred_val_h))
                mae_val = mean_absolute_error(y_val_h, y_pred_val_h)
                mape_val = mean_absolute_percentage_error(y_val_h, y_pred_val_h)
                r2_val = r2_score(y_val_h, y_pred_val_h)
                
                # Test metrics
                rmse_test = np.sqrt(mean_squared_error(y_test_h, y_pred_test_h))
                mae_test = mean_absolute_error(y_test_h, y_pred_test_h)
                mape_test = mean_absolute_percentage_error(y_test_h, y_pred_test_h)
                r2_test = r2_score(y_test_h, y_pred_test_h)
                
                # OOB score from the specific estimator
                oob_r2 = model.estimators_[h_idx].oob_score_ if self.oob_score else np.nan
                
                print(f"  Horizon t+{h}: Train R²={r2_train:.4f} | Val RMSE={rmse_val:.2f}, MAE={mae_val:.2f}, R²={r2_val:.4f} | Test RMSE={rmse_test:.2f}, MAE={mae_test:.2f}, R²={r2_test:.4f}")
                
                results_list.append({
                    'fold': fold + 1,
                    'horizon': h,
                    'strategy': 'RF_MIMO',
                    'n_estimators': self.n_estimators,
                    'max_features': self.max_features,
                    'train_rmse': rmse_train,
                    'train_mae': mae_train,
                    'train_r2': r2_train,
                    'val_rmse': rmse_val,
                    'val_mae': mae_val,
                    'val_mape': mape_val,
                    'val_r2': r2_val,
                    'test_rmse': rmse_test,
                    'test_mae': mae_test,
                    'test_mape': mape_test,
                    'test_r2': r2_test,
                    'oob_r2': oob_r2,
                    'n_features': len(self.features),
                    'model': model if h_idx == 0 else None,
                    'y_pred_val': y_pred_val_h,
                    'y_pred_test': y_pred_test_h,
                    'y_true_val': y_val_h,
                    'y_true_test': y_test_h
                })
        
        self.cv_results = results_list
        return pd.DataFrame(results_list)


# In[ ]:


# ========================================
# TRAIN RANDOM FOREST MODELS
# ========================================

print("\n" + "="*100)
print("PHASE 2: RANDOM FOREST REGRESSION")
print("="*100)

# ----------------------------------------
# TRAIN RF DIRECT STRATEGY
# ----------------------------------------

rf_direct_strategy = RandomForestDirectStrategy(
    horizons=HORIZONS,
    features=effective_features,
    n_estimators=200,           # 200 trees - good balance of performance and speed
    max_features='sqrt',         # sqrt(35) ≈ 6 features per split
    max_depth=None,              # No limit - let trees grow fully (3M samples = low overfitting risk)
    min_samples_split=10,        # Require 10 samples to split (slight regularization)
    min_samples_leaf=5,          # Require 5 samples per leaf (prevents tiny, noisy leaves)
    random_state=42,
    n_jobs=-1,                   # Use all CPU cores
    oob_score=True               # Calculate out-of-bag error
)

print("\n" + "-"*100)
print("Training Random Forest Direct Strategy...")
print("-"*100)

rf_direct_results = rf_direct_strategy.fit_and_evaluate(
    X_scaled=X_full_scaled,
    y_dict=y_full_dict,
    cv_splits=cv_splits
)

print("\n✓ Random Forest Direct Strategy complete")
rf_direct_summary = rf_direct_results.groupby('horizon')[['test_rmse', 'test_mae', 'test_r2']].agg(['mean', 'std'])
print("\nDirect Strategy Summary (Test Set):")
print(rf_direct_summary)

# ----------------------------------------
# TRAIN RF MIMO STRATEGY
# ----------------------------------------

rf_mimo_strategy = RandomForestMIMOStrategy(
    horizons=HORIZONS,
    features=effective_features,
    n_estimators=200,
    max_features='sqrt',
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    oob_score=True
)

print("\n" + "-"*100)
print("Training Random Forest MIMO Strategy...")
print("-"*100)

rf_mimo_results = rf_mimo_strategy.fit_and_evaluate(
    X_scaled=X_full_scaled,
    y_dict=y_full_dict,
    cv_splits=cv_splits
)

print("\n✓ Random Forest MIMO Strategy complete")
rf_mimo_summary = rf_mimo_results.groupby('horizon')[['test_rmse', 'test_mae', 'test_r2']].agg(['mean', 'std'])
print("\nMIMO Strategy Summary (Test Set):")
print(rf_mimo_summary)

# ----------------------------------------
# SAVE RESULTS
# ----------------------------------------

# Combine all Random Forest results
all_rf_results = pd.concat([
    rf_direct_results,
    rf_mimo_results
], ignore_index=True)

all_rf_results.to_csv('random_forest_results.csv', index=False)
print("\n✓ Random Forest results saved to: random_forest_results.csv")


# ## Comparing the Linear Regression and Random Forest Models 

# In[ ]:


# ========================================
# COMPARE ALL MODELS
# ========================================

# Combine Linear Regression results (from your existing code)
# with Random Forest results

all_models_results = pd.concat([
    direct_results.assign(model_family='Linear'),      # Your existing linear direct results
    mimo_results.assign(model_family='Linear'),        # Your existing linear MIMO results
    rf_direct_results.assign(model_family='RandomForest'),
    rf_mimo_results.assign(model_family='RandomForest')
], ignore_index=True)

# Comprehensive comparison
print("\n" + "="*100)
print("COMPREHENSIVE MODEL COMPARISON: LINEAR REGRESSION vs RANDOM FOREST")
print("="*100)

comparison = all_models_results.groupby(['model_family', 'strategy', 'horizon']).agg({
    'test_rmse': ['mean', 'std'],
    'test_mae': ['mean', 'std'],
    'test_r2': ['mean', 'std']
}).round(4)

print(comparison)

# Statistical significance testing (Paired t-test)
from scipy import stats

print("\n" + "="*100)
print("STATISTICAL SIGNIFICANCE: RF vs Linear (Paired t-test per horizon)")
print("="*100)

for horizon in HORIZONS:
    # Direct strategy comparison
    linear_direct_rmse = direct_results[direct_results['horizon']==horizon]['test_rmse'].values
    rf_direct_rmse = rf_direct_results[rf_direct_results['horizon']==horizon]['test_rmse'].values
    
    t_stat, p_value = stats.ttest_rel(linear_direct_rmse, rf_direct_rmse)
    improvement = ((linear_direct_rmse.mean() - rf_direct_rmse.mean()) / linear_direct_rmse.mean() * 100)
    
    print(f"\nHorizon t+{horizon}h (Direct Strategy):")
    print(f"  Linear RMSE: {linear_direct_rmse.mean():.2f} ± {linear_direct_rmse.std():.2f}")
    print(f"  RF RMSE:     {rf_direct_rmse.mean():.2f} ± {rf_direct_rmse.std():.2f}")
    print(f"  Improvement: {improvement:.2f}%")
    print(f"  p-value:     {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'}")

# Save comprehensive results
all_models_results.to_csv('comprehensive_model_comparison.csv', index=False)
print("\n✓ Comprehensive results saved to: comprehensive_model_comparison.csv")


# ## Feature Importance Analysis ( Random Forest)

# In[ ]:


# ========================================
# FEATURE IMPORTANCE ANALYSIS
# ========================================

import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "="*100)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*100)

# Set publication-quality plotting style
plt.rcParams.update({
    'axes.labelsize': 10,
    'font.size': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.titlesize': 11,
    'figure.figsize': [14, 10],
    'savefig.dpi': 300
})

# Create 2x2 subplot for each horizon
fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
axes = axes.flatten()

for idx, horizon in enumerate(HORIZONS):
    importance_df = rf_direct_strategy.get_feature_importance(horizon, top_n=15)
    
    ax = axes[idx]
    
    # Horizontal bar plot
    y_pos = np.arange(len(importance_df))
    ax.barh(y_pos, importance_df['mean'].values, xerr=importance_df['std'].values,
            color='#2171b5', alpha=0.8, capsize=3)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Gini Importance', fontweight='bold', fontsize=10)
    ax.set_title(f'Top 15 Features for t+{horizon}h Forecast', 
                 fontweight='bold', fontsize=11)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add importance values as text
    for i, (mean_val, std_val) in enumerate(zip(importance_df['mean'].values, 
                                                  importance_df['std'].values)):
        ax.text(mean_val + std_val + 0.001, i, f'{mean_val:.3f}', 
                va='center', fontsize=8)

plt.savefig('rf_feature_importance_by_horizon.pdf', bbox_inches='tight')
plt.savefig('rf_feature_importance_by_horizon.png', bbox_inches='tight')
plt.show()

print("\n✓ Feature importance plots saved")

# Print detailed importance for each horizon
for horizon in HORIZONS:
    print(f"\n{'='*80}")
    print(f"Feature Importance for t+{horizon}h Forecast")
    print(f"{'='*80}")
    importance = rf_direct_strategy.get_feature_importance(horizon, top_n=20)
    print(importance.to_string())


# ## Prediction Visualization ( For Random Forest)

# In[ ]:


# ========================================
# PREDICTION QUALITY VISUALIZATION
# ========================================

def plot_predictions_vs_actual(results_df, strategy_name, horizons, sample_size=1000):
    """
    Create scatter plots of predicted vs actual values for each horizon.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    axes = axes.flatten()
    
    for idx, horizon in enumerate(horizons):
        ax = axes[idx]
        
        # Get predictions from last fold
        fold_data = results_df[(results_df['horizon'] == horizon) & 
                               (results_df['strategy'] == strategy_name)].iloc[-1]
        
        y_true = fold_data['y_true_test']
        y_pred = fold_data['y_pred_test']
        
        # Sample for visualization if dataset is large
        if len(y_true) > sample_size:
            indices = np.random.choice(len(y_true), sample_size, replace=False)
            y_true = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
            y_pred = y_pred[indices]
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.3, s=10, color='#2171b5')
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                label='Perfect Prediction')
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        ax.set_xlabel('Actual Traffic Flow (vehicles/hour)', fontweight='bold')
        ax.set_ylabel('Predicted Traffic Flow (vehicles/hour)', fontweight='bold')
        ax.set_title(f't+{horizon}h Forecast: RMSE={rmse:.1f}, MAE={mae:.1f}, R²={r2:.3f}',
                     fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)
    
    plt.savefig(f'{strategy_name}_predictions_vs_actual.pdf', bbox_inches='tight')
    plt.savefig(f'{strategy_name}_predictions_vs_actual.png', bbox_inches='tight')
    plt.show()

# Plot for RF Direct strategy
plot_predictions_vs_actual(rf_direct_results, 'RF_Direct', HORIZONS)

# Plot for RF MIMO strategy
plot_predictions_vs_actual(rf_mimo_results, 'RF_MIMO', HORIZONS)

print("\n✓ Prediction visualization complete")


# ## Performance Comparison Visualization (Linear Regression And Random Forest)

# In[ ]:


# ========================================
# COMPREHENSIVE PERFORMANCE COMPARISON
# ========================================

def plot_model_comparison(all_results_df, horizons):
    """
    Create side-by-side comparison of RMSE and MAE across all models and horizons.
    """
    # Prepare data
    summary = all_results_df.groupby(['model_family', 'strategy', 'horizon']).agg({
        'test_rmse': ['mean', 'std'],
        'test_mae': ['mean', 'std']
    }).reset_index()
    
    summary.columns = ['model_family', 'strategy', 'horizon', 
                       'rmse_mean', 'rmse_std', 'mae_mean', 'mae_std']
    
    # Create model labels
    summary['model'] = summary['model_family'] + ' ' + summary['strategy'].str.replace('RF_', '').str.replace('_', ' ')
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    
    # RMSE comparison
    ax1 = axes[0]
    x = np.arange(len(horizons))
    width = 0.2
    
    models = summary['model'].unique()
    colors = ['#084594', '#2171b5', '#6baed6', '#bdd7e7']
    
    for i, model in enumerate(models):
        model_data = summary[summary['model'] == model].sort_values('horizon')
        ax1.bar(x + i*width, model_data['rmse_mean'], width, 
                yerr=model_data['rmse_std'], label=model, 
                color=colors[i], capsize=4)
    
    ax1.set_xlabel('Forecast Horizon (hours)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('RMSE (vehicles/hour)', fontweight='bold', fontsize=12)
    ax1.set_title('(a) Root Mean Squared Error Comparison', fontweight='bold', fontsize=13)
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(horizons)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # MAE comparison
    ax2 = axes[1]
    for i, model in enumerate(models):
        model_data = summary[summary['model'] == model].sort_values('horizon')
        ax2.bar(x + i*width, model_data['mae_mean'], width,
                yerr=model_data['mae_std'], label=model,
                color=colors[i], capsize=4)
    
    ax2.set_xlabel('Forecast Horizon (hours)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('MAE (vehicles/hour)', fontweight='bold', fontsize=12)
    ax2.set_title('(b) Mean Absolute Error Comparison', fontweight='bold', fontsize=13)
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(horizons)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.savefig('comprehensive_model_comparison_plot.pdf', bbox_inches='tight')
    plt.savefig('comprehensive_model_comparison_plot.png', bbox_inches='tight')
    plt.show()

plot_model_comparison(all_models_results, HORIZONS)

print("\n✓ Comprehensive comparison visualization complete")


# ## HYPERPARAMETER TUNING: DIRECT (One Model Per Horizon)

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

def tune_random_forest_hyperparameters(X_train, y_train, cv_splits=3):
    param_grid = {
        'n_estimators': [100, 200],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    base_rf = RandomForestRegressor(random_state=42, n_jobs=-1, oob_score=True)
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_estimator_

# Run for each horizon (column in Y_train):
direct_best_params = []
direct_best_estimators = []
for h in range(Y_train.shape[1]):
    print(f"Tuning DIRECT for horizon {h+1}")
    params, estimator = tune_random_forest_hyperparameters(X_train, Y_train[:, h])
    direct_best_params.append(params)
    direct_best_estimators.append(estimator)


# ## HYPERPARAMETER TUNING: MIMO (ONE Model for All Horizons)

# In[ ]:


from sklearn.multioutput import MultiOutputRegressor

def tune_mimo_hyperparameters(X_train, Y_train, cv_splits=3):
    param_grid = {
        'estimator__n_estimators': [100, 200],
        'estimator__max_features': ['sqrt', 'log2'],
        'estimator__max_depth': [10, 20],
        'estimator__min_samples_split': [2, 5],
        'estimator__min_samples_leaf': [1, 2]
    }
    base_rf = RandomForestRegressor(random_state=42, n_jobs=-1, oob_score=True)
    model = MultiOutputRegressor(base_rf)
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, Y_train)
    return grid_search.best_params_, grid_search.best_estimator_

print("Tuning MIMO for all horizons together ...")
mimo_best_params, mimo_best_estimator = tune_mimo_hyperparameters(X_train, Y_train)


# ## REFIT RANDOM FOREST MODELS WITH BEST PARAMETERS

# ## For DIRECT (One Model Per Horizon)

# In[ ]:


direct_final_models = []
for h in range(Y_train.shape[1]):
    print(f"Refitting DIRECT for horizon {h+1}")
    model = RandomForestRegressor(**direct_best_params[h], random_state=42, n_jobs=-1)
    model.fit(X_train, Y_train[:, h])
    direct_final_models.append(model)
# Predictions
DIRECT_preds = np.column_stack([m.predict(X_test) for m in direct_final_models])


# ## For MIMO (ONE Model for All Horizons)

# In[ ]:


base_rf = RandomForestRegressor(random_state=42, n_jobs=-1)
MIMO_final_model = MultiOutputRegressor(base_rf.set_params(**{
    k.replace('estimator__', ''): v for k,v in mimo_best_params.items()
}))
MIMO_final_model.fit(X_train, Y_train)
MIMO_preds = MIMO_final_model.predict(X_test)


# ## EVALUATE MODELS (MAE, RMSE, MAPE, $R^2$ Per Horizon) - Thus Direct and MIMO (Random Forest)

# In[ ]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values'))
    mape = np.mean(np.abs((y_true - y_pred)/y_true), axis=0) * 100
    r2 = r2_score(y_true, y_pred, multioutput='raw_values')
    return mae, rmse, mape, r2

direct_mae, direct_rmse, direct_mape, direct_r2 = calc_metrics(Y_test, DIRECT_preds)
mimo_mae, mimo_rmse, mimo_mape, mimo_r2 = calc_metrics(Y_test, MIMO_preds)


# ##  FEATURE IMPORTANCE – DIRECT & MIMO (Random Forest)

# In[ ]:


import pandas as pd
# For DIRECT: Feature importance per horizon
importances = np.array([model.feature_importances_ for model in direct_final_models])
mean_importance = importances.mean(axis=0)
std_importance = importances.std(axis=0)
# For MIMO
mimo_importances = np.mean([e.feature_importances_ for e in MIMO_final_model.estimators_], axis=0)
# Save for reporting
fi_df = pd.DataFrame({'feature': feature_names,
                     'direct_mean': mean_importance,
                     'direct_std': std_importance,
                     'mimo_mean': mimo_importances})
fi_df = fi_df.sort_values('direct_mean', ascending=False)
fi_df.to_csv('feature_importances_compare.csv', index=False)


# # ADVANCED INTERPRETATION WITH SHAP ( Random Forest)

# ## For Direct (One Model Per Horizon)

# In[ ]:


import shap
# Compute SHAP for one or more horizons
for h, model in enumerate(direct_final_models):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print(f"SHAP summary for DIRECT model at horizon {h+1}")
    shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
    plt.savefig(f'shap_summary_direct_h{h+1}.pdf', dpi=600)
    plt.close()


# ## For MIMO

# In[ ]:


import shap
import matplotlib.pyplot as plt

features = list(X_train.columns)  
for h, estimator in enumerate(MIMO_final_model.estimators_):
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_test)
    print(f"SHAP summary for MIMO output {h+1}")
    shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
    plt.savefig(f'shap_summary_mimo_out{h+1}.pdf', dpi=600)
    plt.close()


# ## STATISTICAL COMPARISON (Diebold-Mariano Test)

# In[ ]:


from scipy.stats import norm

def diebold_mariano(loss1, loss2):
    d = loss1 - loss2
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    DM_stat = mean_d / np.sqrt(var_d / len(d))
    p_value = 2 * norm.sf(np.abs(DM_stat))
    return DM_stat, p_value

dm_results = []
for h in range(Y_test.shape[1]):
    loss_direct = np.abs(Y_test[:, h] - DIRECT_preds[:, h])
    loss_mimo = np.abs(Y_test[:, h] - MIMO_preds[:, h])
    dm_stat, p_val = diebold_mariano(loss_direct, loss_mimo)
    dm_results.append({'horizon': h, 'DM_stat': dm_stat, 'p_value': p_val})
dm_df = pd.DataFrame(dm_results)
dm_df.to_csv('diebold_mariano_results.csv', index=False)


# ## PREDICTION INTERVALS (Quantile Regression Forest)

# In[ ]:


from skgarden import RandomForestQuantileRegressor
qrf = RandomForestQuantileRegressor(**direct_best_params[0])  
qrf.fit(X_train, Y_train[:, 0])
lower_q = qrf.predict(X_test, quantile=5)
upper_q = qrf.predict(X_test, quantile=95)
coverage = np.mean((Y_test[:, 0] >= lower_q) & (Y_test[:, 0] <= upper_q))
print(f'Coverage for 90% PI: {coverage:.2f}')


# ## PLOTS

# ### Mean Decrease in Impurity (MDI, Global Importance)

# In[ ]:


import pandas as pd

importances = np.array([model.feature_importances_ for model in direct_final_models])
mean_importance = importances.mean(axis=0)
std_importance = importances.std(axis=0)
features = list(X_train.columns) 
feature_df = pd.DataFrame({
    'feature': features,
    'mean_importance': mean_importance,
    'std_importance': std_importance
})
feature_df = feature_df.sort_values('mean_importance', ascending=False)
feature_df.to_csv('direct_feature_importance.csv', index=False)


# ## Feature Importance with Error Bars

# In[ ]:


import matplotlib.pyplot as plt
# Bar plot with error bars (mean ± std) for DIRECT
plt.figure(figsize=(10,5))
plt.barh(feature_df['feature'].iloc[:15], feature_df['mean_importance'].iloc[:15],
         xerr=feature_df['std_importance'].iloc[:15], color='skyblue', edgecolor='black')
plt.xlabel('Mean MDI Importance')
plt.title('Top 15 Features: DIRECT (mean ± std)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('direct_feature_importance_top15.pdf', dpi=600)
plt.close()


# ## Residual Analysis: Histogram & Q-Q Plot

# In[ ]:


import scipy.stats as stats
for h in range(Y_test.shape[1]):
    # Histogram of residuals
    resids = Y_test[:, h] - DIRECT_preds[:, h]
    plt.figure(figsize=(8,4))
    plt.hist(resids, bins=40, color='lightgray', edgecolor='black')
    plt.title(f'Residual Histogram: DIRECT Horizon {h+1}')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'direct_residual_hist_h{h+1}.pdf', dpi=600)
    plt.close()

    # Q-Q plot for normality
    plt.figure(figsize=(6,6))
    stats.probplot(resids, dist='norm', plot=plt)
    plt.title(f'Q-Q Plot: DIRECT Residuals Horizon {h+1}')
    plt.tight_layout()
    plt.savefig(f'direct_qqplot_h{h+1}.pdf', dpi=600)
    plt.close()


# # LSTM for Spatio-Temporal Traffic Forecasting

# ### Fundamental Limitations of Current Models

# Linear Regression and Random Forest treat time series as independent observations. While we have engineered excellent features (temporal lags, spatial neighbors, rolling statistics), these models fundamentally lack sequential memory.
# 
# ### Why not RNN ?
# The vanishing gradient problem prevents simple Recurrent Neural Networks (RNNs) from learning long-term dependencies. Gradients exponentially decay when backpropagating through many time steps, making it impossible for vanilla RNNs to learn that "traffic 24 hours ago influences traffic now

# ### Why LSTM?

# How LSTM Solves These Problems:
# LSTM introduces a cell state that acts as a memory highway with controlled information flow through three gates:
# 
# 1.Forget Gate: Decides what to discard from memory
# 2.Input Gate: Decides what new information to store
# 3.Output Gate: Decides what to output from the memory
# 
# LSTM is the theoretically appropriate model for temporal, non-linear, long-range spatio-temporal forecasting.
# 
# NB:The cell state update through addition rather than multiplication

# This additive structure allows gradients to flow backward through time without vanishing, enabling the network to learn dependencies across 24+ timesteps.

# The current data is tabular (samples × features). LSTM requires 3D input: (samples, timesteps, features). Hence we will first have to transform tabular data to sequences.

# In[ ]:


def create_sequences(X, y, lookback):
    """
    Returns 3D array (n_samples, lookback, n_features) and aligned y.
    """
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X.iloc[i:i+lookback].values)
        ys.append(y.iloc[i+lookback])
    return np.stack(Xs), np.array(ys)


# ## LSTM Models ( Direct and MIMO)

# Direct LSTM: Train a separate model for each horizon (4 models per fold)

# MIMO LSTM: Train a single model per fold with multi-output for all 4 horizons.

# NB: This explanation for MIMO and direct applies also for all the models that have been already built (Linear Regression (Elastic Net) and Random Forest)

# In[ ]:


def build_direct_lstm(input_shape, lstm_units=64, dropout=0.3):
    model = Sequential([
        LSTM(lstm_units, input_shape=input_shape, return_sequences=True),
        Dropout(dropout),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

def build_mimo_lstm(input_shape, output_dim, lstm_units=64, dropout=0.3):
    model = Sequential([
        LSTM(lstm_units, input_shape=input_shape, return_sequences=True),
        Dropout(dropout),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout),
        Dense(output_dim)
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model


# ## Cross-Validation LSTM For Training (Direct and MIMO, with Hyperparameter Tuning)

# In[ ]:


def lstm_grid_search(X_full_scaled, y_full_dict, cv_splits, effective_features, HORIZONS,
                     lookback=24, mode='direct', param_grid=None, max_trials=8, seed=42):
    """
    - mode: 'direct' or 'mimo'
    - param_grid: dict {param: [vals]}, e.g. {'lstm_units':[32,64], 'dropout':[0.2,0.3], ...}
    - max_trials: number of random parameter combos to try
    Returns: all_results (fold/horizon/metrics grid)
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    all_results = []
    param_grid = param_grid or {
        'lstm_units': [64, 128],
        'dropout': [0.2, 0.3],
        'batch_size': [32, 64],
        'lookback': [24]
    }
    # build combos
    from itertools import product
    keys, grids = zip(*param_grid.items())
    combos = random.sample(list(product(*grids)), min(max_trials, np.prod([len(g) for g in grids])))
    
    for fold, (tr_idx, val_idx, te_idx) in enumerate(cv_splits):
        print(f"\nFold {fold+1}/{len(cv_splits)}")
        X_tr = X_full_scaled.iloc[tr_idx][effective_features]
        X_val = X_full_scaled.iloc[val_idx][effective_features]
        X_te = X_full_scaled.iloc[te_idx][effective_features]
        
        # Target(s)
        if mode=='direct':
            fold_results = []
            for h in HORIZONS:
                y_tr = y_full_dict[h].iloc[tr_idx]
                y_val = y_full_dict[h].iloc[val_idx]
                y_te = y_full_dict[h].iloc[te_idx]
                best_val_rmse = 1e12
                best_params, best_model = None, None
                for settings in combos:
                    config = dict(zip(keys, settings))
                    # Create sequences
                    Xtr_seq, ytr_seq = create_sequences(X_tr, y_tr, config['lookback'])
                    Xval_seq, yval_seq = create_sequences(X_val, y_val, config['lookback'])
                    # Build model
                    model = build_direct_lstm(
                        input_shape=(config['lookback'], Xtr_seq.shape[2]),
                        lstm_units=config['lstm_units'],
                        dropout=config['dropout'])
                    es = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')
                    model.fit(Xtr_seq, ytr_seq, 
                        validation_data=(Xval_seq, yval_seq),
                        epochs=30, batch_size=config['batch_size'],
                        verbose=0, callbacks=[es])
                    val_pred = model.predict(Xval_seq).flatten()
                    val_rmse = np.sqrt(mean_squared_error(yval_seq, val_pred))
                    if val_rmse < best_val_rmse:
                        best_val_rmse, best_params, best_model = val_rmse, config, model
                # Test on test set
                Xte_seq, yte_seq = create_sequences(X_te, y_te, best_params['lookback'])
                te_pred = best_model.predict(Xte_seq).flatten()
                te_rmse = np.sqrt(mean_squared_error(yte_seq, te_pred))
                te_mae = mean_absolute_error(yte_seq, te_pred)
                te_r2 = r2_score(yte_seq, te_pred)
                fold_results.append({'fold': fold+1, 'horizon': h, 
                                    'rmse': te_rmse, 'mae': te_mae, 'r2': te_r2,
                                    'params': best_params})
            all_results.extend(fold_results)
        else:  # MIMO
            # Y targets as multi-class
            Y_tr = np.column_stack([y_full_dict[h].iloc[tr_idx] for h in HORIZONS])
            Y_val = np.column_stack([y_full_dict[h].iloc[val_idx] for h in HORIZONS])
            Y_te = np.column_stack([y_full_dict[h].iloc[te_idx] for h in HORIZONS])
            best_val_rmse = 1e12
            best_params, best_model = None, None
            for settings in combos:
                config = dict(zip(keys, settings))
                Xtr_seq, Ytr_seq = create_sequences(X_tr, pd.DataFrame(Y_tr), config['lookback'])
                Xval_seq, Yval_seq = create_sequences(X_val, pd.DataFrame(Y_val), config['lookback'])
                model = build_mimo_lstm(input_shape=(config['lookback'], Xtr_seq.shape[2]),
                                       output_dim=Ytr_seq.shape[1],
                                       lstm_units=config['lstm_units'],
                                       dropout=config['dropout'])
                es = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')
                model.fit(Xtr_seq, Ytr_seq, 
                      validation_data=(Xval_seq, Yval_seq),
                      epochs=30, batch_size=config['batch_size'],
                      verbose=0, callbacks=[es])
                val_pred = model.predict(Xval_seq)
                val_rmse = np.mean([np.sqrt(mean_squared_error(Yval_seq[:,i], val_pred[:,i])) for i in range(Ytr_seq.shape[1])])
                if val_rmse < best_val_rmse:
                    best_val_rmse, best_params, best_model = val_rmse, config, model
            Xte_seq, Yte_seq = create_sequences(X_te, pd.DataFrame(Y_te), best_params['lookback'])
            te_pred = best_model.predict(Xte_seq)
            for i,h in enumerate(HORIZONS):
                te_rmse = np.sqrt(mean_squared_error(Yte_seq[:,i], te_pred[:,i]))
                te_mae = mean_absolute_error(Yte_seq[:,i], te_pred[:,i])
                te_r2 = r2_score(Yte_seq[:,i], te_pred[:,i])
                all_results.append({'fold': fold+1, 'horizon': h, 'rmse': te_rmse, 'mae': te_mae, 'r2': te_r2,
                                   'params': best_params})
    return pd.DataFrame(all_results)


# ## EXECUTION

# In[ ]:


# PARAM_GRID for tuning; you can expand grid for more thorough search
param_grid = {
    'lstm_units': [64, 128],
    'dropout': [0.2, 0.3],
    'batch_size': [32, 64],
    'lookback': [24]
}

# --- DIRECT LSTM ---
lstm_direct_results = lstm_grid_search(X_full_scaled, y_full_dict, cv_splits, effective_features, HORIZONS,
                                  lookback=24, mode='direct', param_grid=param_grid, max_trials=8)
print(lstm_direct_results.groupby('horizon')[['rmse', 'mae', 'r2']].mean())

# --- MIMO LSTM ---
lstm_mimo_results = lstm_grid_search(X_full_scaled, y_full_dict, cv_splits, effective_features, HORIZONS,
                                lookback=24, mode='mimo', param_grid=param_grid, max_trials=8)
print(lstm_mimo_results.groupby('horizon')[['rmse', 'mae', 'r2']].mean())


# ## PLOTS FOR THE LSTM MODEL 

# ## Forecast vs. Actual Time Series (Overlay)

# In[ ]:


import matplotlib.pyplot as plt

def plot_forecast_vs_actual(y_true, y_pred, horizon, strategy, idx_range=None):
    """
    y_true: true target values
    y_pred: LSTM predictions (Direct: array, MIMO: column of array)
    horizon: int (hours ahead)
    strategy: 'Direct' or 'MIMO'
    idx_range: slice for plotting subset (optional)
    """
    if idx_range is None:
        idx_range = slice(0, min(200, len(y_true)))  # show only a sample window for clarity
    plt.figure(figsize=(10, 4))
    plt.plot(y_true[idx_range], label='Actual', color='black')
    plt.plot(y_pred[idx_range], label='Predicted', color='red', alpha=0.7)
    plt.xlabel('Time Index')
    plt.ylabel('Traffic Flow')
    plt.title(f'{strategy} LSTM: Actual vs Predicted (t+{horizon}h)')
    plt.legend()
    plt.tight_layout()
    plt.show()


# ## FOr DIRECT 

# In[ ]:


plot_forecast_vs_actual(y_true_24h, direct_pred_24h, horizon=24, strategy='Direct')


# ## FOR MIMO 

# In[ ]:


plot_forecast_vs_actual(y_true_24h, mimo_pred[:,1], horizon=24, strategy='MIMO')  # index 1 for 24h


# ## Error Distribution Boxplot

# In[ ]:


import seaborn as sns
import pandas as pd

def plot_error_distribution(results_df, error_metric='rmse'):
    """
    results_df: DataFrame with columns 'strategy', 'horizon', 'fold', error_metric
    """
    plt.figure(figsize=(8,4))
    sns.boxplot(x='horizon', y=error_metric, hue='strategy', data=results_df)
    plt.title(f'{error_metric.upper()} Distribution by Horizon and LSTM Strategy')
    plt.xlabel('Forecast Horizon (hours)')
    plt.ylabel(error_metric.upper())
    plt.tight_layout()
    plt.show()


# In[ ]:


import seaborn as sns
import pandas as pd

def plot_error_distribution(results_df, error_metric='mae'):
    """
    results_df: DataFrame with columns 'strategy', 'horizon', 'fold', error_metric
    """
    plt.figure(figsize=(8,4))
    sns.boxplot(x='horizon', y=error_metric, hue='strategy', data=results_df)
    plt.title(f'{error_metric.upper()} Distribution by Horizon and LSTM Strategy')
    plt.xlabel('Forecast Horizon (hours)')
    plt.ylabel(error_metric.upper())
    plt.tight_layout()
    plt.show()


# In[ ]:


results_df = pd.DataFrame({
    'strategy': all_strategies,  
    'horizon': all_horizons,
    'fold': all_folds,         
    'rmse': all_rmses,
    'mae': all_maes
})
plot_error_distribution(results_df, error_metric='rmse')


# In[ ]:


results_df = pd.DataFrame({
    'strategy': all_strategies,  
    'horizon': all_horizons,     
    'fold': all_folds,          
    'rmse': all_rmses,
    'mae': all_maes
})
plot_error_distribution(results_df, error_metric='mae')


# ## Actual vs. Predicted Scatter Plot

# In[ ]:


def plot_actual_vs_pred(y_true, y_pred, horizon, strategy):
    plt.figure(figsize=(5,5))
    plt.scatter(y_true, y_pred, alpha=0.3, color='royalblue')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=2)  # perfect
    plt.xlabel('Actual Traffic')
    plt.ylabel('Predicted Traffic')
    plt.title(f'{strategy} LSTM Actual vs Predicted (t+{horizon}h)')
    plt.tight_layout()
    plt.show()


# In[ ]:


## For Direct 


# In[ ]:


plot_actual_vs_pred(y_true_48h, direct_pred_48h, horizon=48, strategy='Direct')


# In[ ]:


## For MIMO


# In[ ]:


plot_actual_vs_pred(y_true_48h, mimo_pred[:,2], horizon=48, strategy='MIMO')


# ## Feature Importance for LSTM – Permutation

# In[ ]:


import numpy as np

def permutation_feature_importance(lstm_model, X_test_seq, y_test, feature_names):
    base_pred = lstm_model.predict(X_test_seq).flatten()
    base_rmse = np.sqrt(mean_squared_error(y_test, base_pred))
    importances = []
    for i, fname in enumerate(feature_names):
        X_test_perm = np.copy(X_test_seq)
        # Permute the feature in all time steps
        flat_feat = X_test_perm[:,:,i].flatten()
        np.random.shuffle(flat_feat)
        X_test_perm[:,:,i] = flat_feat.reshape(X_test_perm[:,:,i].shape)
        perm_pred = lstm_model.predict(X_test_perm).flatten()
        perm_rmse = np.sqrt(mean_squared_error(y_test, perm_pred))
        importances.append(perm_rmse - base_rmse)
    imp_df = pd.DataFrame({'feature':feature_names, 'importance':importances})
    imp_df = imp_df.sort_values('importance', ascending=False)
    plt.figure(figsize=(9,5))
    plt.barh(imp_df['feature'], imp_df['importance'], color='darkslateblue')
    plt.xlabel('RMSE Increase (Importance)')
    plt.title('Permutation Feature Importance – LSTM')
    plt.tight_layout()
    plt.show()
    return imp_df


# ## For Direct

# In[ ]:


imp_df_24h = permutation_feature_importance(direct_lstm_model_24h, X_test_seq_24h, y_test_24h, effective_features)


# ## FOr MIMO

# In[ ]:


imp_df_24h = permutation_feature_importance(mimo_lstm_model, X_test_seq, y_test_24h, effective_features)


# ## SHAP for Direct and MIMO LSTM

# ### SHAP on Direct LSTM

# In[ ]:


# import shap
# import numpy as np

# # Example for Direct LSTM, horizon = 24h
# direct_model_24h = ...          # Your trained direct LSTM model for 24h
# X_test_seq_24h = ...            # Shape: (samples, lookback, features)
# feature_names = effective_features  # List of feature names used

# # Use sufficiently small batch (DeepExplainer memory)
# background = X_test_seq_24h[:50]   # Subsample for background (8)
# target = X_test_seq_24h[:100]      # Subsample for explanations
# explainer = shap.DeepExplainer(direct_model_24h, background)
# shap_values = explainer.shap_values(target)
# # shap_values[0]: (samples, lookback, features)

# # Aggregate over time dimension for each feature
# mean_shap = np.mean(np.abs(shap_values[0]), axis=(0,1)) # (features,)
# shap.summary_plot(shap_values[0], features=target, feature_names=feature_names)

# # print top features
# imp_df = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_shap})
# imp_df.sort_values('mean_abs_shap', ascending=False).head(10)


# ## SHAP on MIMO LSTM

# In[ ]:


# # MIMO model outputs (samples, 4) for 4 horizons
# mimo_model = ...                # Your trained MIMO LSTM model
# X_test_seq_mimo = ...           # Shape: (samples, lookback, features)
# feature_names = effective_features
# background = X_test_seq_mimo[:50]
# target = X_test_seq_mimo[:100]
# explainer = shap.DeepExplainer(mimo_model, background)
# shap_values = explainer.shap_values(target)  # List of arrays, one per output node

# # For each horizon (output node), plot feature importance
# horizon_indices = {12: 0, 24: 1, 48: 2, 72: 3}  # Map output node index to horizon
# for horizon, idx in horizon_indices.items():
#     print(f"\nMIMO LSTM SHAP for t+{horizon}h:")
#     mean_shap = np.mean(np.abs(shap_values[idx]), axis=(0,1))
#     shap.summary_plot(shap_values[idx], features=target, feature_names=feature_names, show=False)
#     plt.title(f'SHAP MIMO LSTM: Feature Effects on t+{horizon}h Output')
#     plt.tight_layout()
#     plt.show()
#     # Optionally print top features
#     imp_df = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_shap})
#     print(imp_df.sort_values('mean_abs_shap', ascending=False).head(10))


# In[ ]:





# ## Why Did we Consider Linear Regression 

# The first model we considered:ELASTIC NET (Linear Regression with Regularization)

# | Advantage                | Why It Matters                                                |
# | ------------------------ | ------------------------------------------------------------- |
# | Interpretability         | Coefficients show exact contribution of each feature          |
# | Transparency             | Easy to explain to non-technical stakeholders (city planners) |
# | Computational efficiency | Fast to train, easy to deploy in real-time systems            |
# | Regularization           | Handles multicollinearity (common in traffic data)            |
# | Established theory       | Well-understood statistical properties                        |

# ## Why Did we consider Random Forest?

# The next model we consider is the random forest model (Ensemble Tree-Based Method)

# 1. Captures Non-Linear Relationships
# 2.  Robustness to Outliers

# | Traffic Pattern         | Linear Model             | Random Forest                 |
# | ----------------------- | ------------------------ | ----------------------------- |
# | Rush hour non-linearity | Assumes linear effect    |  Captures threshold effects   |
# | Spatial interactions    | Manual interaction terms | Auto-discovers patterns       |
# | Weekend vs weekday      | Separate dummy variables | Learns complex differences    |
# 

# # Appendix - Key Notes

#  Why Direct Strategy is Optimal : **Direct strategy** involves training four independent linear regression models, each optimized for a specific forecast horizon. This approach offers several critical advantages that align with publication standards:
# 
# **Elimination of error accumulation**: Unlike recursive methods where prediction errors compound across time steps, direct models make independent forecasts at each horizon. This is particularly crucial for your 72-hour horizon, where recursive approaches can experience dramatic performance degradation. Research on traffic flow prediction demonstrates that recursive strategies suffer from accumulated errors that significantly impact long-term forecasts.
# 
# **Horizon-specific optimization**: Each model can identify and leverage different feature relationships relevant to its specific time horizon. For instance, immediate traffic conditions may be most predictive for 12-hour forecasts, while broader patterns like day-of-week effects become more important for 72-hour predictions. This horizon-specific feature utilization is impossible with a single recursive model.
# 
# **Superior long-term performance**: Extensive empirical studies comparing forecasting strategies demonstrate that direct methods consistently outperform recursive approaches for longer horizons (48-72 hours), which are critical for your traffic prediction application. A comprehensive review of 111 time series found that multiple-output strategies (direct and MIMO) delivered the best performance across various forecast horizons.
# 
# **Publication value**: Training separate models allows you to provide detailed, horizon-specific analyses that reviewers and readers find valuable. You can report which features matter most at each horizon, how prediction uncertainty evolves, and where your models succeed or struggle.
# The MIMO Alternative: A Necessary ComparisonWhile recommending the Direct strategy as primary, you should also implement a **Multiple-Input Multiple-Output (MIMO)** approach for comparison. MIMO uses a single linear regression model that simultaneously predicts all four horizons, capturing inter-horizon dependencies.
# 
# **Key advantages of MIMO**:
# 
# - **Structured predictions**: MIMO can learn that consecutive forecast horizons are related, potentially improving forecast consistency
# - **Computational efficiency**: One model instead of four reduces training time and storage requirements
# - **Captures temporal dynamics**: Can model how uncertainty propagates across horizons
# 
# **Why MIMO alone isn't sufficient**: Research shows that while MIMO performs well, it typically doesn't match the horizon-specific accuracy of direct models, particularly at longer horizons where specialized optimization matters most. However, including MIMO in your study provides a comprehensive comparison that strengthens your publication by demonstrating you evaluated multiple approaches.
# 
# Avoid Pure Recursive Strategy for Multi-Day HorizonsThe recursive (iterated) strategy training a model for one-step-ahead (y+1) and iterating to reach 72 hours—is **not recommended** for your multi-day forecast horizons. While recursive methods excel at very short-term predictions (1-6 hours), they face critical limitations for your 48-72 hour targets:
# 
# - **Error propagation**: Each prediction becomes input for the next, causing errors to compound exponentially over 72 iterations
# - **Accumulating bias**: Small biases in short-term predictions magnify into large systematic errors at longer horizons
# - **Loss of feature information**: Recursive predictions replace actual features with predicted values, degrading information quality
# 
# Traffic-specific studies confirm these limitations. Research on LSTM models for traffic flow prediction found that recursive strategies deteriorate significantly beyond 12-hour horizons, while direct approaches maintain stable performance.
# 
# Feature Set Evaluation: 31 Features with 3 Million ObservationsYour team's concern about whether 31 engineered features is "good enough" can be addressed with statistical rigor and comparison to standards in the literature.
# 
# Statistical Assessment: Excellent Feature-to-Sample RatioYour dataset configuration provides exceptional statistical properties for linear regression:
# 
# **Feature-to-sample ratio**: With 3 million observations and 35 features, you have approximately **85,714 observations per feature**. Standard statistical guidelines recommend minimum 10-20 observations per feature for reliable parameter estimation. Your ratio exceeds this by more than 4,000-fold, placing you in an extremely safe zone against overfitting.
# 
# **Separate models consideration**: Even training four independent models (one per horizon) maintains this excellent ratio, as each model still has access to all 3 million training samples. The Direct strategy doesn't partition your data—it only changes the target variable (y+12, y+24, y+48, y+72) while using the same feature matrix.]
# 
# **Comparison to published work**: Recent traffic forecasting publications typically work with 10-50 features on datasets ranging from 10,000 to 500,000 observations. Your 3 million observations with 35 features is exceptionally well-powered for robust inference.
# 
# Feature Selection and Validation: The Path to Publication ExcellenceRather than viewing 35 features as fixed, frame your publication around **validating your feature engineering process** through systematic selection and comparison:
# 
# **Recommended approach**:
# 
# 1. **Start with all 35 features**: Establish baseline performance with your complete engineered feature set across all horizons
# 
# 2. **Apply regularization-based selection**: Implement LASSO (L1) or Elastic Net regularization to perform automatic feature selection while preventing overfitting. This is particularly important for publication as it:
#    - Provides statistical justification for feature inclusion
#    - Identifies which features contribute most to each horizon
#    - Demonstrates methodological rigor that reviewers expect
# 
# 3. **Calculate feature importance**: Use SHAP values or permutation importance to quantify each feature's contribution to predictions at different horizons. This analysis often reveals that different features dominate at different forecast horizons, providing valuable insights for your publication.
# 
# 4. **Test reduced feature sets**: Systematically evaluate models with 25, 20, and 15 features selected by LASSO to demonstrate whether your full feature set is justified or if a parsimonious model performs comparably. This sensitivity analysis strengthens your paper by showing robustness
# **Regularization for publication quality**: Even though overfitting is unlikely given your sample size, applying Elastic Net regularization (combining L1 and L2 penalties) provides multiple benefits:
# - Automatic handling of correlated features common in traffic data
# - Shrinkage of coefficients for more stable predictions
# - Publication credibility through demonstrated best practices
# - Feature selection that can be reported and interpreted
# 
# Recent studies demonstrate that regularization methods substantially improve forecasting performance even with adequate sample sizes, particularly in high-dimensional time series contexts.
# 
#  Feature Engineering Best Practices for Traffic ForecastingYour feature engineering process should be thoroughly documented for publication. State-of-the-art traffic forecasting research emphasizes several feature categories that should be represented in your 35 features:
# 
# **Temporal features**: Hour of day, day of week, month, holiday indicators, season—these capture periodic patterns fundamental to traffic
# 
# **Lag features**: Historical traffic values at relevant lags (e.g., t-1, t-24, t-168 for week-over-week patterns) that provide autoregressive information
# 
# **Spatial features**: If applicable, features capturing traffic at adjacent locations or network connectivity, as spatial correlations are strong in traffic networks
# 
# **External factors**: Weather conditions, special events, construction indicators if available and relevant
# 
# **Derived features**: Rolling statistics (moving averages, standard deviations), trend indicators, and interaction terms between temporal and spatial variables
# 
# Documenting how your 35 features span these categories and comparing to the literature provides context that strengthens your publication.[18][22][24]
# 
# Validation Methodology: Publication StandardsFor a publication-quality study, your validation approach must be rigorous and align with time series forecasting best practices.
# 
# Time Series Cross-ValidationStandard k-fold cross-validation violates temporal ordering and leads to data leakage in time series contexts. Instead, implement **expanding window or rolling window cross-validation**:
# 
# **Expanding window approach** (recommended):
# - Start with initial training period (e.g., first year of data)
# - Train models and forecast next 72 hours
# - Expand training window to include just-forecasted period
# - Repeat, progressively growing training set
# - Provides realistic simulation of operational forecasting
# 
# **Rolling window approach** (alternative):
# - Maintain fixed training window size
# - Slide window forward through time
# - Useful if recent patterns are most relevant
# 
# With 3 million hourly observations (approximately 342 years if continuous, or more realistically, data from many locations), you can implement multiple validation folds that provide robust performance estimates.
# 
#  Performance Metrics for PublicationReport multiple metrics to provide comprehensive evaluation:
# 
# **Primary metrics**:
# - **RMSE** (Root Mean Square Error): Penalizes large errors, standard in regression
# - **MAE** (Mean Absolute Error): More robust to outliers, easier interpretation
# - **MAPE** (Mean Absolute Percentage Error): Enables comparison across different traffic scales
# 
# **Statistical testing**:
# - **Diebold-Mariano test**: Formally test whether performance differences between Direct and MIMO strategies are statistically significant
# - **Confidence intervals**: Report 95% confidence intervals for all metrics across validation folds
# 
# **Horizon-specific analysis**:
# - Report metrics separately for each forecast horizon (12h, 24h, 48h, 72h)
# - Demonstrate how prediction accuracy degrades (or remains stable) with horizon length
# - This analysis provides valuable insights for practitioners and validates your methodological choices
# 
# -**Direct strategy implementation**:
# - Train four separate linear regression models with Elastic Net regularization (α=0.5 mixing parameter, λ via cross-validation)
# - Model 1: Predict traffic at t+12 hours
# - Model 2: Predict traffic at t+24 hours  
# - Model 3: Predict traffic at t+48 hours
# - Model 4: Predict traffic at t+72 hours
# 
# **MIMO implementation**:
# - Train single linear regression model with 4 outputs, one for each horizon
# - Use same regularization approach for fair comparison
# 
# **Recursive baseline**:
# - Train one-step-ahead model, iterate to 72 steps
# - Include as baseline to demonstrate why direct approach is superior for multi-day horizons
# 
# Feature Selection and Engineering Validation **Systematic feature analysis**:
# - Apply LASSO path analysis to identify feature selection thresholds
# - Calculate SHAP values to quantify feature importance for each horizon
# - Test reduced feature sets (30, 25, 20, 15 features) to establish robustness
# - Document which features are most important for short-term vs long-term predictions
# 
# This phase provides rich content for your publication's methodology and results sections, demonstrating thorough investigation beyond simple model comparison.
# 
# Optimization and Comparison **Hyperparameter tuning**:
# - Grid search over regularization parameters (α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}, λ via cross-validation)
# - Select optimal parameters based on validation performance
# 
# **Comprehensive validation**:
# - Implement expanding window cross-validation with multiple folds
# - Calculate all metrics (RMSE, MAE, MAPE) for each horizon
# - Perform Diebold-Mariano statistical tests comparing Direct vs MIMO
# 
#  Analysis and Insights **Error analysis**:
# - Analyze where and when models make large errors
# - Investigate whether errors are concentrated in specific conditions (e.g., rush hours, weekends, holidays)
# - Examine prediction uncertainty evolution across horizons
# 
# **Feature stability**:
# - Assess whether the same features remain important across validation folds
# - Report which features are consistently selected by LASSO across horizons
# 
# Documentation and Submission **Manuscript preparation**:
# - Introduction: Motivate multi-horizon forecasting for traffic management
# - Literature review: Position your work relative to existing studies on forecasting strategies
# - Methodology: Thoroughly document Direct, MIMO, and Recursive approaches; feature engineering; regularization; validation procedure
# - Results: Present comprehensive comparisons with statistical testing, feature importance analysis, error analysis by horizon
# - Discussion: Interpret findings, provide recommendations for practitioners, acknowledge limitations
# - Conclusion: Summarize contributions and future research directions
# 
# **Target journals**: Transportation Research Part C, IEEE Transactions on Intelligent Transportation Systems, Transportation Research Record, or similar high-impact venues publishing quantitative traffic forecasting research.
# 
# Expected Findings and Publication ContributionsBased on the literature and your problem characteristics, your study will likely find:
# 
# 1. **Direct strategy outperforms Recursive**, especially at 48-72 hour horizons, due to error accumulation in recursive approach
# 
# 2. **Direct vs MIMO comparison is context-dependent**: Direct likely achieves slightly better accuracy at each specific horizon, while MIMO provides competitive performance with computational efficiency
# 
# 3. **Feature importance varies by horizon**: Short-term forecasts (12h) rely heavily on recent traffic patterns and hour-of-day, while long-term forecasts (72h) emphasize day-of-week and seasonal patterns
# 
# 4. **35 features are justified**: Your engineered features provide value beyond simple autoregressive models, particularly for longer horizons where complex patterns matter
# 
# 5. **Regularization improves generalization**: Even with 3M samples, Elastic Net provides more stable and interpretable models than unregularized regression
# 
# These findings will constitute **original empirical contributions** to traffic forecasting literature, particularly the Direct vs MIMO comparison in the context of simple linear models with extensive feature engineering.
# 
# Summary of Key Recommendations**Primary strategy**: Implement Direct approach with four separate linear regression models, one per forecast horizon (12h, 24h, 48h, 72h)
# 
# **Comparison strategy**: Also train MIMO and Recursive models to provide comprehensive empirical comparison
# **Feature set**: Your 35 features with 3M observations is excellent; validate through regularization-based selection and importance analysis
# 
# **Regularization**: Apply Elastic Net to all models to prevent overfitting, enable feature selection, and demonstrate methodological rigor
# 
# **Validation**: Use expanding window time series cross-validation with multiple metrics (RMSE, MAE, MAPE) and statistical testing (Diebold-Mariano)
# 
# **Publication angle**: Frame as comprehensive comparison of multi-horizon strategies for traffic forecasting with linear models, featuring thorough feature engineering validation
# 
# This approach positions your team to produce publication-quality research that makes meaningful contributions to both transportation forecasting methodology and practical traffic prediction applications. The combination of rigorous statistical methods, comprehensive strategy comparison, and large-scale empirical validation will meet the standards of top-tier journals in the field.
# 
# 

# ## Visually how the Expanding Window Cross Validation Looks Like for Time Series Forecasting 
# 
# |---- Train ----|--- Forecast 1 --->|
# 
# 
# |--------- Train (expanded) --------|--- Forecast 2 --->|
# 
# 
# |------------------- Train ---------|--- Forecast 3 --->|
# 
# ... and so on.
# 

# ## Thought Process 

# In[ ]:


PHASE 1: BASELINE (Linear Model) ✓ DONE
├── Train Elastic Net Direct ✓
├── Train Elastic Net MIMO ✓
├── Calculate feature importance ✓
└── Winner: Direct strategy for Elastic Net ✓

PHASE 2: ADVANCED MODELS (Use FULL feature set for all)
├── Random Forest
│   ├── Train RF Direct (35 features)
│   ├── Train RF MIMO (35 features)
│   ├── Calculate RF's OWN feature importance
│   └── Compare: Direct vs MIMO for RF
│
│
└──  LSTM/Neural Network
    ├── Train LSTM Direct (35 features)
    ├── Train LSTM MIMO (35 features)
    └── Compare: Direct vs MIMO for LSTM


    

PHASE 3: ANALYSIS & DISCUSSION
├── Compare ALL models on SAME feature set
├── Show feature importance for EACH model separately
├── Discuss: "Why does RF prefer spatial features while Linear prefers temporal?"
└── Justify: "We used full feature set for fair comparison (Guyon & Elisseeff, 2003)"

PHASE 4: SENSITIVITY ANALYSIS (Optional - Supplementary)
├── Test each model with its OWN top 15 features
├── Show: Performance with 35 vs 15 features
└── Conclusion: "Full feature set necessary" or "Top 15 sufficient"


# # Random Forest Regression: Mathematical and Algorithmic Details
# 
# 
# ## 1. Introduction
# Random Forest regression is a nonparametric ensemble method. It builds randomized decision trees and combines their predictions for robust, accurate regression, particularly in high-dimensional, nonlinear, and noisy data settings.
# 
# 
# ## 2. Mathematical Foundations of Regression Trees
# 
# ### 2.1 Regression Tree Model
# Given training data $$ \{ (\mathbf{x}_i, y_i) \}_{i=1}^n $$, $$ \mathbf{x}_i \in \mathbb{R}^p $$, $$ y_i \in \mathbb{R} $$:
# 
# - The tree recursively partitions the predictor space into axis-aligned regions $$ R_1, \ldots, R_M $$.
# - Prediction function:
# 
# $$
#     f_T(\mathbf{x}) = \sum_{m=1}^{M} c_m \; \mathbb{I}(\mathbf{x} \in R_m)
# $$
# 
# where:  
# $$ c_m = \frac{1}{|R_m|} \sum_{\mathbf{x}_i \in R_m} y_i $$
# 
# ### 2.2 Recursive Binary Splitting
# At each node, the split $(j^*, s^*)$ selects the feature $j^*$ and value $s^*$ that minimize total impurity:
# 
# $$
# (j^*,s^*) = \underset{j,s}{\arg\min}\Bigg[ \sum_{\mathbf{x}_i \in R_L(j, s)} (y_i - \bar{y}_L)^2 + \sum_{\mathbf{x}_i \in R_R(j, s)} (y_i - \bar{y}_R)^2 \Bigg]
# $$
# 
# where:
# - $$ R_L(j, s) = \{ \mathbf{x}: x_j \leq s \} $$
# - $$ R_R(j, s) = \{ \mathbf{x}: x_j > s \} $$
# 
# Splitting continues until a rule is met (minimum samples per leaf, maximum depth, or node purity).
# 
# 
# ## 3. Bagging: Bootstrap Aggregation
# - Reduces variance compared to single decision trees.
# - For $$ B $$ rounds, sample with replacement to get bootstrap datasets $\mathcal{D}_b$.
# - Fit regression tree $T_b$ to each sample.
# - Aggregate by averaging predictions:
# 
# $$ \hat{f}_{\mathrm{bag}}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x}) $$
# 
# Variance of the bagged estimator:
# $$\mathrm{Var}(\hat{f}_{\mathrm{bag}}) = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2 $$
# where $$ \sigma^2 $$ is tree variance, $$ \rho $$ is tree-to-tree correlation. Lower $$ \rho $$ leads to greater variance reduction.
# 
# 
# ## 4. Random Forest: Feature Randomization
# - At each split, randomly select $$ m_{\text{try}} \ll p $$ predictors instead of all $$ p $$.
# - Deep, unpruned trees are grown for each bootstrapped dataset.
# 
# **Algorithm Steps:**
# 1. For each of $B$ trees:
#     - Draw bootstrap sample $\mathcal{D}_b$.
#     - At each split, select $m_{\text{try}}$ random predictors.
#     - Find the best split among those.
# 2. Aggregate predictions:
# $$
# \hat{f}_{\text{RF}}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x}; \Theta_b)
# $$
# where $\Theta_b$ encodes bootstrap sampling and feature selection.
# 
# Default for regression: $m_{\text{try}} = \lfloor p/3 \rfloor$ or $\sqrt{p}$.
# 
# 
# ## 5. Out-of-Bag (OOB) Error Estimation
# On average, 36.8% of training data is "out-of-bag" (OOB) for each tree.
# - The OOB prediction for observation $i$ is
# $$
# \hat{y}_i^{\text{OOB}} = \frac{1}{|\mathcal{B}_i|} \sum_{b \in \mathcal{B}_i} T_b(\mathbf{x}_i)
# $$
# where $\mathcal{B}_i$ is set of trees where $i$ was OOB.
# - The overall OOB mean squared error is
# $$
# \mathrm{MSE}_{\text{OOB}} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i^{\text{OOB}} )^2
# $$
# 
# 
# ## 6. Bias–Variance Decomposition
# Prediction error decomposes as:
# $$
# \mathbb{E}[ (\hat{f}_{\text{RF}}(\mathbf{x}) - f(\mathbf{x}))^2 ] = \text{Bias}^2 + \text{Variance} + \text{Noise}
# $$
# Random Forest maintains low bias but majorly reduces variance by decorrelating trees.
# 
# 
# ## 7. Feature Importance Measures
# 
# ### 7.1 Mean Decrease in Impurity (MDI)
# Total decrease in impurity (e.g., MSE) for each feature over all trees:
# $$
# \text{MDI}(j) = \sum_{\text{splits on } j} \Delta\text{impurity}
# $$
# 
# ### 7.2 Permutation Importance (MDA)
# 1. First estimate baseline OOB error.
# 2 Permute feature j in out of bag data
# 
# Importance(j)=Errorperm(j)−ErrorOOB
# 
# 
# | Step                     | Operation                                     |
# | ------------------------ | --------------------------------------------- |
# | Bootstrap sampling       | Draw (n) samples to form (\mathcal{D}_b)      |
# | Random feature selection | Choose (m_{\text{try}}) out of (p) predictors |
# | Split optimization       | Select split that minimizes squared error     |
# | Tree prediction          | Mean response within leaf region              |
# | Forest prediction        | (\frac{1}{B}\sum_b T_b(\mathbf{x}))           |
# | Out of bag error         | Predict using trees where sample was OOB      |
# | MDI importance           | Sum of impurity reductions                    |
# | Permutation importance   | Increase in error after permuting feature     |
# 
# 

# 

# In[ ]:




