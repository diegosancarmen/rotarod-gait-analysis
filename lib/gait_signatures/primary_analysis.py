import numpy as np
import pandas as pd

def process_csv_to_dataframe_filter(filepath):
    path = str(filepath + ".csv")

    df = pd.read_csv(path).copy()

    col_list = []
    for i in range(0, df.shape[1]):
        col_list.append(
            str(
                str(df.iloc[1, i]) +
                "_" +
                str(df.iloc[2, i])[0]
            )
        )

    df.columns = col_list
    df = df.drop([0, 1, 2])

    # df = df.fillna(df.mean())
    # df = df.astype(float, errors = 'raise')

    # Extract a list of all columns containing "_l"
    l_columns = [col for col in df.columns if col.endswith("_l")]

    # Iterate through the _l columns and apply the condition to corresponding _x and _y columns
    for l_column in l_columns:
        x_column = l_column.replace("_l", "_x")
        y_column = l_column.replace("_l", "_y")

        # Convert the columns to numeric values and replace non-numeric values with NaN
        df[x_column] = pd.to_numeric(df[x_column], errors='coerce')
        df[y_column] = pd.to_numeric(df[y_column], errors='coerce')
        df[l_column] = pd.to_numeric(df[l_column], errors='coerce')

        # Apply the condition
        condition = df[l_column] > 0.1
        df[x_column] = np.where(condition, df[x_column], np.nan)
        df[y_column] = np.where(condition, df[y_column], np.nan)

    return df

def add_distance_columns(df):
    df['leftpaw_x_d'] = (df['tailbase_x'] - df['leftpaw_x']).abs()
    df['rightpaw_x_d'] = (df['tailbase_x'] - df['rightpaw_x']).abs()

def add_rod_columns(df):
    df['rod_y_avg'] = ((df['rodleft_y'] + df['rodright_y']) / 2).mean()
    df['tailbase_rod_y_dist'] = df['rod_y_avg'] - df['tailbase_y']
