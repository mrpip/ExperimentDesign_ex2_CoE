import os
os.chdir(r'C:\Users\Pia\OneDrive\Data Science\experiment_design\ex_2\CoE_dataset')
import xml.etree.ElementTree as et
import pandas as pd
pd.options.display.max_columns = 34
pd.options.display.max_rows = 100


def load_data(data_set):
    # get the columnnames:
    xtree = et.parse(r'Dev_Set\XML\A_Fish_Called_Wanda.xml')
    xroot = xtree.getroot()
    columns = list(xroot.find('movie').keys())
    df = pd.DataFrame(columns=columns)

    for movie_name in os.listdir(data_set + '_Set/XML'):
        xtree = et.parse(data_set + '_Set/XML/' + movie_name)
        xroot = xtree.getroot()
        instance = []
        for element in columns:
            if xroot[0] is not None:  # and element != 'goodforairplane': # feature "good for airplane" is not filled in
                instance.append(xroot.find('movie').get(element))
            else:
                pass
        if len(instance) < len(columns):
            instance.append(movie_name[:-3])
        my_series = pd.Series(instance, index=columns)
        df = df.append(my_series, ignore_index=True)

    df.set_index('title', inplace=True)

    df = df.loc[:,
         ['language', 'year', 'genre', 'country', 'runtime', 'rated', 'metascore', 'imdbRating', 'tomatoUserRating']]

    # make the feature 'runtime' numeric:
    df.runtime = df.runtime.apply(lambda x: x[:-4])
    df.runtime = pd.to_numeric(df.runtime, errors='coerce')

    # bring the entries of 'rated' which were not filled out into a unique shape:
    df.rated.replace(['NOT RATED', 'UNRATED'], 'N/A', inplace=True)

    for feature in ['year', 'metascore', 'imdbRating', 'tomatoUserRating']:
        df.loc[:, feature] = pd.to_numeric(df.loc[:, feature], errors='coerce')

    return df


all_categorical_features = 'language', 'genre', 'country', 'rated'


def get_dummies(df, selected_features=all_categorical_features):
    df_tmp = df.copy()
    for feature in set(selected_features).intersection(all_categorical_features):  # we only get dummy
        # variables for categorical
        # data
        # split the variables with various entries:
        one_hot = df.copy().loc[:, feature].str.split(', ', expand=True).stack()

        one_hot = pd.get_dummies(one_hot, prefix=feature, drop_first=True).groupby(level=0).sum()
        df_tmp = df_tmp.drop(feature, axis=1)
        df_tmp = df_tmp.merge(one_hot, left_index=True, right_index=True)

    return df_tmp
