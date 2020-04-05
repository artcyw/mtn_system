"""
This tool take in dataframe and will simplify the data science process of Cleaning, Exploring, Preprocessing and Modeling.

Require the most common package for Data Science:
pandas
sklearn
matplotlib
seaborn
numpy
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import sklearn.metrics

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet

from sklearn.pipeline import Pipeline

class cleaner:
    """
    A class to clean DataFrame

    ...

    Attributes
    ----------

    df : Access initialize DataFrame

    textClean: Lowercase all text, strip spaces and standardize columns. (Lowercase with spacer replace with underscore)

    naSummary: Generate DataFrame with null values details
    
    """
    def __init__(self, df):
        """
        Parameters
        ----------
        df : Pandas DataFrame

        """
        self.df = df
        null = self.df.isna().sum().sum()
        print(f'Shape: {self.df.shape}, Null values: {null}')
        
    
    @property
    def textClean(self):
        def try_strip(x):
            try:
                return x.strip()
            except:
                return x
            
        self.df = self.df.applymap(lambda x: try_strip(x)) # strip to reduce all spaces and turn to empty cells
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_') 
        self.df = self.df.applymap(lambda x : x.lower() if isinstance(x, str) else x)
    
    @property    
    def naSummary(self):
        na_mean = self.df.isna().mean()[self.df.isna().mean().sort_values() > 0].sort_values()
        na_total = self.df.isna().sum()[self.df.isna().sum().sort_values() > 0].sort_values() 

        na_df = pd.DataFrame(na_total, columns=['total_nulls'])
        na_df['mean_nulls'] = na_mean
        na_df['total_count'] = self.df.isna().count()
        na_df['dtypes'] = self.df.dtypes

        return na_df

class explore:
    '''A class to explore data'''
    def __init__(self, df, target):
        self.df = df
        self.target = target
        
        self.num_features = self.df.select_dtypes(include='number').columns
        
        self.nom_features = self.df.select_dtypes(exclude='number').columns
        
        corr = self.df[self.num_features].corr()[[self.target]] # make correlation table
        corr[self.target] = abs(corr[self.target])
        self.highcor = corr.sort_values(self.target, ascending=False).iloc[1:]
        
        print(f'File shape: {self.df.shape}, Null Values: {self.df.isna().sum().sum()}')
    
    def heatmap(self):
        corr = self.df[self.num_features].corr()[[self.target]]
        plt.figure(figsize=(16,9))
        return sns.heatmap(corr.sort_values(self.target), cmap='coolwarm', annot=True);
    
    def hc_hist(self, num_top_features=10):
        'high correlation of numberical features, default is top 10'
        return self.df[self.highcor.head(num_top_features).index].hist(figsize=(16,12));
    
    def scatter(self, start=0, end=-1):
        'scatter plot for all numerical features. Start and End to index the graphs, if there is too many for better viewing'
        features = self.num_features[start:end]
        fig, ax = plt.subplots(round(len(features) / 3), 3, figsize = (18,12))        
        
        for i, ax in enumerate(fig.axes):
            if i <= len(features) - 1:
                sns.regplot(x=features[i], y=self.df[self.target], data=self.df[features], ax=ax)
        plt.tight_layout();
        
        print(f'graph displayed: {len(features)} out of {len(self.num_features)}')
    
    def countplot(self, start=0, end=-1):
        features = self.nom_features[start:end]
        fig, ax = plt.subplots(round(len(features) / 3), 3, figsize = (18,12))   
        features_used = []
        
        for i, ax in enumerate(fig.axes):
            if i <= len(features) - 1:
                ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
                ax.set_xlabel(ax.get_xlabel(), fontsize=16)
                sns.countplot(x=features[i], data=self.df[features], ax=ax)
                features_used.append(features[i])

        plt.tight_layout()
        
        print(f'graph displayed: {len(features)} out of {len(self.nom_features)}. Features used: {features_used}')

class featureEng:
    'A Feature Engineering class'
    def __init__(self, df, target):
        'Initiate feature eng with a csv_file and target'
        # read in data frame 
        
        self.df = df
        self.target = target
        self.features = self.df.drop(target, axis=1)
        
        self.num_features = self.features.select_dtypes(include='number').columns
        self.nom_features = self.features.select_dtypes(exclude='number').columns
        
        self.dummies = pd.get_dummies(self.df[self.nom_features], drop_first=True) # right now dummying all nominal features
        
        self.base_df = pd.concat([self.dummies, self.df[self.num_features], self.df[self.target]], axis=1)
        
        print(f'File shape: {self.df.shape}, Null Values: {self.df.isna().sum().sum()}')
        
    def feature_dropping(self, drop_features):
        # pass in list of feature to drop
        
        self.base_df = self.base_df.drop([drop_features], axis=1)
        print('features dropped!')
            
    def poly(self, features):
        'Transform selected features with Polynomial, add back to base DataFrame. Provided a List of features.'
        
        # update base dataframe with polys 
        
        
        X = self.base_df[features]

        poly = PolynomialFeatures(include_bias=False)
        X_poly = poly.fit_transform(X)
        p_features = poly.get_feature_names(features)
        poly_features = pd.DataFrame(X_poly, columns=p_features)

        
        self.base_df = self.base_df.drop(features, axis=1) # remove features used
        
        self.base_df = pd.concat([self.base_df, poly_features], axis=1) # add poly features back to base df
        
        print('DataFrame updated with Polynomial features!')        
        
    def num_features_score(self, model=LinearRegression):
        self.model = model()
        
        s_val_score = []
        s_test_score = []
        s_train_score = []
        s_features = []
        
        for feature in self.num_features:
            X = self.df[[feature]]
            y = self.df[self.target]
        
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=32) 
            self.model.fit(X_train, y_train)
            
            val_score = cross_val_score(self.model, X_train, y_train, cv=5).mean()
            test_score = self.model.score(X_test, y_test)
            train_score = self.model.score(X_train, y_train)
            
            s_features.append(feature)
            s_val_score.append(val_score)
            s_train_score.append(train_score)
            s_test_score.append(test_score)
            
            summary = {'feature' : s_features, 'val_score' : s_val_score, 
                       'train_score' : s_train_score, 'test_score' : s_test_score}
        
        return pd.DataFrame(data=summary).sort_values('val_score', ascending=False)

class lr_model:
    'Modeling Class, default to LinearRegression'
    def __init__(self, df, target, pipe_steps):
        self.df = df
        self.target = self.df[target]
        self.features = self.df.drop(target, axis=1)
        
        self.num_features = self.features.select_dtypes(include='number')
        self.nom_features = self.features.select_dtypes(exclude='number')
        
        self.pipe = Pipeline(pipe_steps)
        
        self.summary = pd.DataFrame({'random_state' : [], 'val_score': [], 'train_score' : [], 'test_score' : []})
        
    def test_models(self, run_time=3):
        'Run Models X amount of time with different random state'
        for i in np.random.choice(100, run_time, replace=False):
            seed = i
            X = self.features
            y = self.target

            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed) 
            self.pipe.fit(X_train, y_train)

            val_score = round(cross_val_score(self.pipe, X_train, y_train, cv=5).mean(), 2)
            test_score = round(self.pipe.score(X_test, y_test), 2)
            train_score = round(self.pipe.score(X_train, y_train), 2)

            self.summary = self.summary.append({'random_state' : i, 'val_score': val_score, 
                                 'train_score' : train_score, 'test_score' : test_score}, ignore_index=True)
        
        return self.summary
    
    def final_model(self):
        pass
    
    def predictions(self):
        pass
                
        
    def coef_score(self):
        
        X = self.features
        y = self.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8) 
        self.model.fit(X_train, y_train)
        
                    
        val_score = cross_val_score(self.model, X_train, y_train, cv=5).mean()
        test_score = self.model.score(X_test, y_test)
        train_score = self.model.score(X_train, y_train)
        
        
        summary = pd.DataFrame({"coefficients": np.transpose([(round(coef, 2)) for coef in self.model.coef_]), 
                             'avg_feature_value' : self.features.mean(), 
                             'avg_feature__median' : self.features.median()})
        
        summary['avg_change'] = summary['avg_feature_value'] * summary['coefficients']
        summary['count'] = self.features[self.features > 0].count()
        
        return summary.sort_values('avg_change', ascending=False)