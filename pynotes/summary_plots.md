Check missing values

```python
import missingno as msno
msno.matrix(df.sample(100))
msno.bar(df.sample(500))
msno.heatmap(df.sample(500))
msno.dendrogram(df.sample(500))
```

Bar plot percentage of NA values for each variable

```python
g = df[df.columns[df.isna().any(0)]].isna().sum().sort_values(ascending=False) / df.shape[0]
g = g.reset_index().rename(columns={'index': 'column', 0: 'nans'})
sns.barplot(x='nans', y='column', data=g, ax=plt.subplots(1,1,figsize=(12,5))[1], palette='rocket')
plt.xlabel('ratio of nans')
```

`plt.subplots`: bar plots and kdeplots

```python
colcat1 = ['poverty_index', 'race', 'weight', 'height', 'systolic_blood_pressure', 'pulse_pressure']
fig, axs = plt.subplots(3, 2, figsize=(15, 20))
axs = axs.ravel()
for i, col_name in enumerate(colcat1):
    if col_name == 'race': # categorical
        percentage = (raw_df.groupby(['Death'])[col_name]
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('race'))
        sns.barplot(x="race", y="percentage", hue="Death", data=percentage, ax = axs[i])
        # sns.countplot(data=raw_df, x=col_name, hue = 'Death',ax=axs[i])
    else:
        sns.kdeplot(data=raw_df, x=col_name, hue = 'Death', common_norm=False, ax=axs[i])
```

`plt.subplots`: Make 9 figures without using `ravel()`.

```python
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
for i, col_name in enumerate(columns_imp[:18]):
    sns.kdeplot(data=raw_df, x=col_name, hue='pred_mortal', common_norm=False, ax=axs[i//3, i%3])
```

Advance plotting with customized labels, text, title for each figure

```python
fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12, 5))

fontsize = 14

cols = ["MRP", "Year", "Kms_Driven"] 
title_vs_text = ["MRP", "Year", "Kms Driven"]
xlabels = ["MRP (in lakhs)", "Year", "Kilometers driven"]

for i, (ax, col, title_vs, xlabel) in enumerate(
    zip(axes, cols, title_vs_text, xlabels)
):

    ax.scatter(df[col], df["Current_Selling_Price"], alpha=0.4)
    ax.set_title(
        f"Current Selling Price vs. {title_vs}",
        fontsize=fontsize,
    )
    ax.set_xlabel(xlabel, fontsize=fontsize-2)
    ax.set_ylabel(
        "Current selling price (in lakhs)" if i==0 else None, 
        fontsize=fontsize-2
    )
    ax.grid(":", alpha=0.4)
    
    if col=="Year":
        min_year = df[col].min()
        max_year = df[col].max()
        ax.set_xticks(np.arange(min_year, max_year+1, 3))

plt.tight_layout()
plt.show()
```

Advanced EDA plotting. Histogram hued by gender.

```python
gendermap = {0:"Women", 1:"Men"}

fig, ax = plt.subplots(1,1, figsize=(10,6))
for i in gendermap.keys():
    ax.hist(
        df2[df2.gender==i].income,
        log=True,
        label = f"Gender type = {gendermap[i]}", 
        alpha=0.4,
        bins = 10,
        density=True,
        edgecolor="w",
    )
    ax.legend(loc="best")
    ax.set_xlabel("Income ($)", fontsize=12)
    ax.set_ylabel("Distribution (Log scale)", fontsize=12)

plt.title("Distribution of income based on gender", fontsize=14)
plt.grid(":", alpha=0.4)
plt.tight_layout()
plt.show()
```

Advanced EDA plotting. Barplot of categorical variables vs. mean continuous outcome. Using `groupby` and `mean`.

```python
fig, axes = plt.subplots(1,2, sharey=True, figsize =(10,4))
df2.groupby(["complexion"]).mean().income.plot(
    kind="bar", alpha=0.8, ax=axes[0]
)
axes[0].set_xlabel("Complexion", fontsize=12)
axes[0].set_title("Mean income by skin complexion", fontsize=14)
axes[0].set_xticklabels(
    ["Very fair","Fair","Wheatish","Medium","Dark"], rotation=0
)
plt.grid(":", alpha=0.4)

df2.groupby(["eating"]).mean().income.plot(kind="bar", alpha=0.8, ax=axes[1])
axes[1].set_xticklabels(
    ["No preference","Jain","Veg","Veg + eggs ","Non-veg"], rotation=0
)
axes[1].set_xlabel("Food eating preference", fontsize=12)
axes[1].set_title("Mean income by eating type", fontsize=14)

for ax in axes:
    ax.set_ylabel("Mean income ($)", fontsize=12)
    ax.grid(":", alpha=0.4)

plt.tight_layout()
plt.show()
```

Scatter matrix for quantitative variables

```python
attr_list = [
    "income",
    "age",
    "height",
    "bmi",
    "status",
    "complexion",
    "education",
]

scatter = pd.plotting.scatter_matrix(
    df2[attr_list], alpha=0.2, figsize=(12,8)
)
for ax in scatter.ravel():
    ax.set_xlabel(ax.get_xlabel(), rotation = 45, fontsize=12)
    ax.set_ylabel(ax.get_ylabel(), rotation = 45, fontsize=12)
plt.suptitle(
    "Scatter matrix of SimplyMarry.com quantitative and ordinal attributes",
    fontsize=18,
    y=0.95,
)
plt.show()
```

Statistical visualization with mean lines and confidence interval lines

```python
fig, ax = plt.subplots(2,2, sharex=False, figsize = (12,10))

ax = ax.ravel()

for i in range(4):
    betavals = bootstrap[:,i]
    betavals.sort()

    x1 = np.percentile(betavals,2.5)
    x2 = np.percentile(betavals,97.5)
    xbar = np.mean(betavals)

    ax[i].hist(bootstrap[:,i], alpha=0.3)

    ax[i].axvline(
        xbar,
        color="k",
        linestyle="-",
        alpha=1,
        label="$\\bar{{\\beta}} = {:.3f}$".format(xbar),
    )
    ax[i].axvline(
        x1,
        color="k",
        linestyle="--",
        alpha=0.8,
        label="95% bounds\n$[{:.3f}, {:.3f}]$".format(x1, x2),
    )
    ax[i].axvline(x2, color="k", linestyle="--", alpha=0.8)
    ax[i].axvline(
        0,
        color="tab:red",
        linestyle=":",
        alpha=1,
        label="$\\beta=0$",
    )

    ax[i].set_ylabel("Distribution",fontsize=12)
    ax[i].set_xlabel("Coefficient",fontsize=12)
    ax[i].grid(":", alpha=0.4)
    ax[i].set_title(f"$\\beta_{i}$", fontsize=14)
    ax[i].legend(fontsize=12, framealpha=1)
    
#plt.xticks(fontsize=20)
fig.suptitle(
    "$95\%$ confidence interval of bootstrapped $\\beta_i$ "
    "coefficients for degree-{} polynomial model".format(best_deg),
    fontsize=16,
)
plt.tight_layout()
plt.show()

print(
    f"\nThe coefficient values of the degree-{best_deg} "
    "polynomial regression model fit in Question "
    "2.1 (without bootstrapping) were:\n"
)

```

Complex MSE plot

```python
def mse_plot( var_list,train_score,val_score,val_std=None,
    title=None,x_label=None,loc="best",log_mse=False,log_xscale=False):

    fig, ax = plt.subplots(figsize=(15, 8))
    
    ax.plot(var_list,train_score,"o--",label="Training",linewidth=2,alpha=0.4)
    ax.plot(var_list,val_score,"^-",label="Validation",markeredgecolor="k",linewidth=2,alpha=0.7)

    if val_std is not None:
        ax.fill_between(
            var_list, 
            np.clip(np.array(val_score)-val_std, a_min=0, a_max=None),
            np.array(val_score)+val_std,
            color="tab:orange",
            alpha=0.2,
            label = "Validation +/-1 standard deviation")

    if log_xscale:
        ax.set_xscale("log")
    
    if log_mse:
        ax.set_yscale("log")
        ax.set_ylabel("$MSE$ (log scaled)", fontsize=12)
    else:
        ax.set_ylabel("$MSE$", fontsize=12)

        
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_title(title, fontsize= 14)
    ax.legend(loc=loc, fontsize =12)
    ax.set_xticks(var_list)
    plt.grid(":", alpha=0.4)
    plt.tight_layout()
    plt.show()
    
    # usage
    x_label = "Degree of polynomial"
title = ("10-fold cross-validated mean train and validation $MSE$ results")

mse_plot(
    var_list=list(range(1, max_degree+1)),
    train_score=train_score,
    val_score=validation_score,
    val_std=validation_std,
    title=title,
    x_label=x_label,
    loc=2,
    log_mse=False,
)

x_label = "Degree of polynomial"
title = ("10-fold cross-validated mean train and validation $MSE$ (log-scaled)")

mse_plot(
    var_list=list(range(1, max_degree+1)),
    train_score=train_score,
    val_score=validation_score,
    val_std=validation_std,
    title=title,
    x_label=x_label,
    loc=2,
    log_mse=True,
)

best_deg = validation_score.index(min(validation_score))+1

print(
    f"\nThe best model has a degree of {best_deg} "
    f"with a training MSE of {train_score[best_deg-1]:.4f}"
    f" and a validation MSE of {min(validation_score):.4f}"
)
```

Visualize cross validation results

```python
def plot_cv_results(
    depths, cvmeans, cvstds, train_scores, title, limit_y=False, show_legend=True,
):

    plt.figure(figsize=(9, 4.5))
    plt.plot(
        depths,cvmeans,
        "^-",
        label="Mean validation",
        markeredgecolor="k",
        color="tab:orange",
        alpha=0.7,
        linewidth=2,
    )
    plt.fill_between(
        depths,
        cvmeans - 2*cvstds,
        cvmeans + 2*cvstds,
        color="tab:orange",
        alpha=0.3,
        label="Validation +/-2 standard deviations",
    )
    
    if limit_y:
        ylim = plt.ylim()
        plt.ylim(ylim)
    
    plt.plot(
        depths,
        train_scores,
        "o--",
        label="Training",
        color="tab:blue",
        alpha=0.4,
        linewidth=2,
    )

    if show_legend:
        plt.legend(fontsize=12)
    
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Maximum tree depth", fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(depths)
    plt.grid(":", alpha=0.4)
    plt.tight_layout()
    plt.show()
    
# plot
depths = list(range(1, 21))
cv = 5

cvmeans, cvstds, train_scores = calc_meanstd(
    X_train, y_train, depths, cv
)

cvmeans = np.array(cvmeans)
cvstds = np.array(cvstds)

title = (
    "Decision tree cross-validation accuracy results by "
    "maximum tree depth"
)
plot_cv_results(
    depths,
    cvmeans,
    cvstds,
    train_scores,
    title,
    limit_y=False,
    show_legend=True,
)

title = (
    "Detailed view of results to illustrate validation trend"
)
plot_cv_results(
    depths,
    cvmeans,
    cvstds,
    train_scores,
    title,
    limit_y=True,
    show_legend=False,
)
```

Plotly 

```python
import plotly.express as px
fig = px.bar(majors["Majors"].value_counts().sort_values().tail(20),
             orientation="h")
fig.update_layout(dict(showlegend=False, xaxis_title="Count", yaxis_title="Major"))
```

```python
px.line(year_sex)
```

```python
dow = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
calls.groupby('Day')['CASENO'].count()[dow].iplot(kind='bar', yTitle='Count')
```

```python
calls['hour_of_day'] = (calls['timestamp'].dt.hour * 60 + calls['timestamp'].dt.minute ) / 60.
py.iplot(ff.create_distplot([calls['hour_of_day']],group_labels=["Hour"],bin_size=1, show_rug=False))
```

```python
px.violin(calls.sort_values("CVDOW"), y="hour_of_day", x="Day", box=True, points="all", hover_name="CVLEGEND")
```

```python
calls['OFFENSE'].value_counts().iplot(kind="bar")
calls['CVLEGEND'].value_counts().iplot(kind="bar")
```

