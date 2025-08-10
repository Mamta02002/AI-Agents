## An Agent that can analyse all kinds of data and create reports and visualizations in minutes which usually takes hours or days by a human.

## Processes/workflow
### Data Preprocessing
1. Removing blank columns and rows
2. Removing unnecessary columns based on entropy, variance and correlation with and without target column if any
3. Handle missing values	Impute or drop missing data depending on strategy
4. Normalize column name, Standardize column names for consistency
5. Convert categorical to numeric, Encode categorical variables (label or one-hot encoding)
6. Detect and remove duplicate rows, Prevent data leakage or redundancy
7. Optional scaling, Normalize numeric features for model compatibility
### Data Analysis
1. Basic insights of data (specific to data)
2. Advanced insights of data (if found any, specific to domain)
3. more variable/points user want to input to analyze
4. built a predictive model if user wants
     a) give suggestions to users for what kind of predictive models can be build on their data (if not sure about need)
     b) build predictive model as user provides its requirements of output
     c) predictive model code will be saved in .py and .ipynb files which can be easily exported and reused by teams to use.
### Data Visualization
1. present visualizations for basic and advanced insights found
     a) producing a report of these insights and charts combined
     b) these visualizations can be deployed on tableau or powerBI or to an excel dashboard (as required by user) -> an advanced feature -- oh hold
     c) different versions of visualizations for the same insights (user can chose best to suit its audience)

### Usage:


