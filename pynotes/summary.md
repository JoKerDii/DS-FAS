`map()`: Transform numerical value to categorical value by adding a string before the number

```python
df = raw_df.copy()
columns_cat1 = ['race']
for col_name in columns_cat1:
    df[col_name] = df[col_name].map(lambda x: f'{col_name}_{x}', na_action='ignore')
    display(df[col_name].head(3))
```

`value_counts()`: Check frequency of levels of categorical variable

```python
display(df['platelets_estimate'].value_counts())
```

`repalce()`: Replace False and True by M and F

```python
cleanup_nums = {"sex_isFemale": {False: 'M', True: 'F'}}
df = df.replace(cleanup_nums)
df['sex_isFemale'].head()
```

`astype()`: Range data type in batches

```python
columns_cat2 = ['platelets_estimate', 'urine_albumin', 'urine_glucose',
                'urine_hematest','race','sex_isFemale']
df[columns_cat2] = df[columns_cat2].astype("category")
df.dtypes
```

`get_dummies()`: One-hot encode categorical variables and concate it with the original data frame

```python
df = pd.get_dummies(df, columns=columns_cat2, prefix=columns_cat2)
# alternatively
df = pd.concat([df] + [pd.get_dummies(df[col_name]) for col_name in columns_cat], axis=1)
```

`isna()`, `drop()`: Calculate the percentage of missing for each variable, and drop those variables with too many missing data.

```python
missing_ratio = df.isna().sum(axis=0) / df.shape[0]
dropcol = df.columns[missing_ratio >= 0.5]
df = df.drop(columns = list(dropcol) + ['sequence_ID'])
df = df.dropna()
```

Create a Boolean variable by inequality and change it to integer type

```python
raw_df['Death'] = (raw_df['y'] > 0).astype('int')
```

`np.where()`: Get coordinate of value we want

```python
col_names = X_train.columns
df_corr = X_train.corr()
itemindex = np.where((df_corr > 0.99) | (df_corr < -0.99))
print("The following pairs of predictor variables have correlation greater than 0.99 or less than -0.99:")
{(col_names[i],col_names[j]): round(df_corr.iloc[i,j],4) for (i,j) in zip(itemindex[0], itemindex[1]) if i > j}
```

Find NA across columns and rows

```python
print(f'Columns with at least one nan value: {df.isna().any(0).sum()}')
print(f'Rows  with at least one nan value: {df.isna().any(1).sum()}')
```

Fill NA with mean

```python
for c in df.select_dtypes('number'):
    df[c] = df[c].fillna(df[c].mean())
```

Fill NA with median

```python
for c in df.select_dtypes('number'):
    df[c] = df[c].fillna(df[c].median())
```

Create a feature mapped from another feature

```python
male_ohe = {'male': 1, 'female': 0, None: None}
temp['is_male'] = temp['D'].apply(lambda x: male_ohe[x]).astype(float)
```

Display most frequent categorical variable levels

```python
majors["Majors"].value_counts().sort_values().tail(20)
```

`pd.pivot_table()`: Group by year, sum up the counts for each sex

```python
year_sex = pd.pivot_table(babynames, 
        index=['Year'], # the row index
        columns=['Sex'], # the column values
        values='Count', # the field(s) to processed in each group
        aggfunc=np.sum,
    )

name_sex = pd.pivot_table(
    babynames, index='Name', columns='Sex', values='Count',
    aggfunc='sum', fill_value=0., margins=True)
```

`apply()`: Apply a function to transform a column

```python
def sex_from_name(name):
    lower_name = name.lower()
    if lower_name not in prop_female.index or prop_female[lower_name] == 0.5:
        return "Unknown"
    elif prop_female[lower_name] > 0.5:
        return "F"
    else:
        return "M"
names['Pred. Sex'] = names['Name'].apply(sex_from_name)
```

Filter rows of a dataset

```python
names[~names["Name"].isin(prop_female.index)]
```

Extract latitude and longitude by regex

```python
calls_lat_lon = (
    calls['Block_Location'].str.replace("\n", "\t") 
    .str.extract(".*\((?P<Lat>\d*\.\d*)\, (?P<Lon>-?\d*\.\d*)\)", expand=True)
)
```

Calculate the fraction of null values in latitude and longitude

```python
(~calls_lat_lon.isnull()).mean()
```

Canonicalization: a process for converting data that has more than one possible representation into a "standard", "normal" or canonical form.

```python
def canonicalize_county(county_name):
    return (
        county_name
        .lower()               # lower case
        .replace(' ', '')      # remove spaces
        .replace('&', 'and')   # replace &
        .replace('.', '')      # remove dot
        .replace('county', '') # remove county
        .replace('parish', '') # remove parish
    )
    
county_and_pop['clean_county'] = county_and_pop['County'].map(canonicalize_county)
county_and_state['clean_county'] = county_and_state['County'].map(canonicalize_county)
```

`split()`: Extract by splitting

```python
pertinent = first.split("[")[1].split(']')[0]
day, month, rest = pertinent.split('/')
year, hour, minute, rest = rest.split(':')
seconds, time_zone = rest.split(' ')
day, month, year, hour, minute, seconds, time_zone
```

Extract date and time by regex

```python
import re
pattern = r'\[(\d+)/(\w+)/(\d+):(\d+):(\d+):(\d+) (.+)\]'
day, month, year, hour, minute, second, time_zone = re.search(pattern, first).groups()
year, month, day, hour, minute, second, time_zone
```

```python
import re
pattern = r'\[(\d+)/(\w+)/(\d+):(\d+):(\d+):(\d+) (.+)\]'
day, month, year, hour, minute, second, time_zone = re.findall(pattern, first)[0]
year, month, day, hour, minute, second, time_zone

# return results as a Series
cols = ["Day", "Month", "Year", "Hour", "Minute", "Second", "Time Zone"]
def log_entry_to_series(line):
    return pd.Series(re.search(pattern, line).groups(), index = cols)
log_entry_to_series(first)
```

Use regular expressions to cut out the extra info in square braces.

```python
vio['clean_desc'] = (vio['desc']
             .str.replace('\s*\[.*\]$', '')
             .str.strip()
             .str.lower())
```

Use regular expressions to assign new features for the presence of various keywords

```python
with_features = (vio
 .assign(is_clean     = vio['clean_desc'].str.contains('clean|sanit'))
 .assign(is_high_risk = vio['clean_desc'].str.contains('high risk'))
 .assign(is_vermin    = vio['clean_desc'].str.contains('vermin'))
 .assign(is_surface   = vio['clean_desc'].str.contains('wall|ceiling|floor|surface'))
 .assign(is_human     = vio['clean_desc'].str.contains('hand|glove|hair|nail'))
 .assign(is_permit    = vio['clean_desc'].str.contains('permit|certif'))
)
```

`melt()`: Transform wide data to long data.

```python
broken_down_by_violation_type = pd.melt(count_features, id_vars=['id', 'date'],
            var_name='feature', value_name='num_vios')
```

Simple Imputation

```python
imputer = SimpleImputer(strategy='mean')
X_train_1 = imputer.fit_transform(X_train)
X_test_1 = imputer.transform(X_test)
```

kNN regression for imputation

```python
scaler = StandardScaler().fit(np.concatenate([X_train, X_test], axis=0))
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn_imputer = KNNImputer(n_neighbors=2)
X_train_2 = knn_imputer.fit_transform(X_train_scaled)
X_test_2 = knn_imputer.transform(X_test_scaled)
```

Simple baseline model

```python
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['y']), df['y'], train_size=0.8)
baseline = RandomForestRegressor()
_ = baseline.fit(X_train, y_train)
y_train_pred = baseline.predict(X_train)
y_test_pred = baseline.predict(X_test)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f'Train RMSE: {np.sqrt(train_mse):.4f}.')
print(f'Test RMSE: {np.sqrt(test_mse):.4f}.')
```

Find important features using random forest, and plot the important features sorted by importance

```python
rf = RandomForestRegressor()
rf.fit(df.drop(columns=['y']), df['y'])
var_imps = rf.feature_importances_
# plot
var_imps = 100.0 * (var_imps / var_imps.max())
sorted_idx = np.argsort(var_imps)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize=(10, 12))
plt.barh(pos, var_imps[sorted_idx], align='center')
plt.yticks(pos, df.drop(columns=['y']).columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
# locate the important features
columns_imp = df.drop(columns=['y']).columns[var_imps > 10]
```

A function performing cross validation at varying tree depths

```python
def calc_meanstd(X_train, y_train, depths, cv):
    cvmeans = []
    cvstds = []
    train_scores = []
    
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth)
        train_scores.append(clf.fit(X_train, y_train).score(X_train, y_train))
        scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=cv)
        cvmeans.append(scores.mean())
        cvstds.append(scores.std())
        
    return cvmeans, cvstds, train_scores
```

[skipped 07_pandas. 10_visualization]
