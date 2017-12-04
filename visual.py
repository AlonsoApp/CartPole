import seaborn as sns
import pickle
import pandas as pd

import os


def to_dataframe(tmp):
    tras = zip(*tmp)
    score_data = list(tras[2])
    avg_data = list(tras[4])
    index = list(tras[0])
    d = {'score': pd.Series(score_data).append(pd.Series(avg_data[100:])),
         'index': pd.Series(index).append(pd.Series(index[100:])),
         'type': pd.Series(['scr']*len(score_data)).append(pd.Series(['avg']*len(avg_data[100:])))}
    df_score = pd.DataFrame(d)
    return df_score


def plot(df_score, name):
    sns.set(style="darkgrid")
    #sns.tsplot(data=tras[2])
    sns_plot = sns.lmplot(x="index", y="score", hue="type", size=8, aspect=2, data=df_score)
    #plt.show()
    sns_plot.savefig("./pickles/reports/"+name+"ng", bbox_inches='tight')


if __name__ == "__main__":

    path = "./pickles/"
    filelist = os.listdir(path)
    for i, file in enumerate(filelist):
        if file.endswith(".p"):
            tuple_list = pickle.load(open(path+file, "rb"))
            df_score = to_dataframe(tuple_list)
            plot(df_score, str(file))
            print(str(i) + "/" + str(len(filelist)) + ": " + str(file))
