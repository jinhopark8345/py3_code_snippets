import pandas as pd
import numpy as np

def get_data():
    df = pd.DataFrame(
        {
            "cate1": ["one", "one", "two", "three"] * 3,
            "cate2": ["A", "B", "C"] * 4,
            "cate3": ["foo", "foo", "foo", "bar", "bar", "bar"] * 2,
            "cate4": np.random.randn(12),
            "cate5": np.random.randn(12),
        }
    )
    print(df)

    #     cat1 cat2 cat3      cat4      cat5
    # 0     one    A  foo  0.431657  1.152855
    # 1     one    B  foo -0.287888  0.760957
    # 2     two    C  foo -0.062445  1.143455
    # 3   three    A  bar  0.412523 -0.303575
    # 4     one    B  bar -0.517554 -0.639041
    # 5     one    C  bar -0.030952  0.350197
    # 6     two    A  foo  0.244239 -0.503637
    # 7   three    B  foo -1.282455 -0.217454
    # 8     one    C  foo  1.584060  0.157564
    # 9     one    A  bar -1.206700  0.372263
    # 10    two    B  bar  0.695477  0.099532
    # 11  three    C  bar -1.355767 -1.721360

    # (Pdb++) df.columns
    # Index(['cat1', 'cat2', 'cat3', 'cat4', 'cat5'], dtype='object')

    # (Pdb++) df.index
    # RangeIndex(start=0, stop=12, step=1)

    return df

def how_to_index(df):
    a = df.loc[df['cate1']=='three', ['cate2', 'cate3','cate4']]
    print(a)
    #     B    C         E
    # 3   A  bar -1.589148
    # 7   B  foo -0.110817
    # 11  C  bar  0.969507

    a = df.loc[df['cate1']=='three', :]
    print(a)
    #         A  B    C         D         E
    # 3   three  A  bar  0.364161  0.753475
    # 7   three  B  foo  0.534567 -0.261629
    # 11  three  C  bar -1.018901  0.091115

# how_to_index(get_data())

def how_to_use_where(df):
    a = df.loc[df['cate1']=='three']
    print(a)
    #         A  B    C         D         E
    # 3   three  A  bar  0.714710 -0.300545
    # 7   three  B  foo  0.545217  1.056457
    # 11  three  C  bar -0.845367 -0.005835

# how_to_use_where(get_data())

def how_to_group_by_column_values(df):
    # a = [df.loc[df['A']==a_val] for a_val in pd.unique(df['A'])]
    # [print(ele) for ele in a]

    #      A  B    C         D         E
    # 0  one  A  foo -1.184755  1.028025
    # 1  one  B  foo -1.749763  1.012021
    # 4  one  B  bar  0.527636  0.772670
    # 5  one  C  bar  0.890923  0.220546
    # 8  one  C  foo -1.376876  1.565086
    # 9  one  A  bar  0.408232 -1.582927
    #       A  B    C         D         E
    # 2   two  C  foo  0.214885  1.287234
    # 6   two  A  foo  0.150951 -1.857709
    # 10  two  B  bar -0.597134  1.016807
    #         A  B    C         D         E
    # 3   three  A  bar  0.530496 -0.516315
    # 7   three  B  foo -0.517710  1.478737
    # 11  three  C  bar -0.591636  0.050275

    groups = df.groupby('cate1')
    for cat1_val in pd.unique(df['cate1']):
        print(groups.get_group(cat1_val))

how_to_group_by_column_values(get_data())
