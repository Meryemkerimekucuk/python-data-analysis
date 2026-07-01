##############################################
# Gelişmiş Fonksiyonel Keşifçi Veri Analizi
###############################################

# 1. Genel Resim
# 2. Kategorik Değişken Analizi
# 3. Sayısal Değişken Analizi
# 4. Hedef Değişken Analizi
# 5. Korelasyon Analizi


#################
# Genel Resim
################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas.conftest import axis

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = sns.load_dataset("titanic")
df.head()

def check_df(dataframe, head=5):
    print("################# SHAPE #################")
    print(dataframe.shape)
    print("################# TYPES #################")
    print(dataframe.dtypes)
    print("################# HEAD #################")
    print(dataframe.head(head))
    print("################# TAIL #################")
    print(dataframe.tail(head))
    print("################# NULL #################")
    print(dataframe.isnull().sum())
    print("################# QUANTILES #################")
    print(dataframe.describe([0,0.05,0.52]))

check_df(df)






###############################
# Kategorik Değişken Analizi
###############################

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = sns.load_dataset("titanic")
df.head()



cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool","str"]]

num_but_col = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int","float"]]

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

cat_cols = cat_cols + num_but_col
cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[cat_cols].nunique()

[col for col in df.columns if col not in cat_cols]


df["survived"].value_counts()

100 * df["survived"].value_counts() / len(df)


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
          "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##############################################")

cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df,col)


import seaborn as sns
import matplotlib.pyplot as plt

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
          "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##############################################")
    if plot:
         sns.countplot(x=dataframe[col_name], data=dataframe)
         plt.show(block=True)

cat_summary(df,"sex", plot=True)

for col in cat_cols:
    if df[col].dtypes == "bool":
        print("-----")

    else:
        cat_summary(df,col, plot=True)


df["adult_male"].astype(int)

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df,col, plot=True)

    else:
        cat_summary(df,col, plot=True)















"""
ÖNERİLMEYEN YÖNTEM
def cat_summary(dataframe, col_name, plot=False):

     if dataframe[col_name].dtypes == "bool":
         dataframe[col_name] = dataframe[col_name].astype(int)

         print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                                "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

         print("##############################################")
         if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

         else:
            dataframe[col_name] = dataframe[col_name].astype(int)

            print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                                    "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
            print("##############################################")
            if plot:
                    sns.countplot(x=dataframe[col_name], data=dataframe)
                    plt.show(block=True)

cat_summary(df, "age", plot=True)
"""






#############################
# Sayısal Değişken Analizi
############################

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df[["age","fare"]].describe().T

num_cols=[col for col in df.columns if df[col].dtypes in ["int","float"]]
num_cols


num_cols = [col for col in num_cols if col not in cat_cols]


def num_summary(dataframe, numeric_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    print(dataframe[numeric_col].describe(quantiles).T)

num_summary(df, "age")

for col in num_cols:
    num_summary(df, col)


def num_summary(dataframe, numeric_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    print(dataframe[numeric_col].describe(quantiles).T)

    if plot:
        dataframe[numeric_col].hist()
        plt.xlabel(numeric_col)
        plt.title(numeric_col)
        plt.show(block=True)

num_summary(df, "age", plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)







############################################################
# Değişkenlerin Yakalanması ve İşlemlerin Genelleştirilmesi
############################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

def grab_col_names(dataframe, cat_th=10, car_th=20):
  """

  Parameters
  ----------
  dataframe: dataframe
        değişken isimleri alınmak istenen dataframedir
  cat_th: int,float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
  car_th: int,float
        kategorik fakat kordinal değişkenler için sınıf eşik değeri

  Returns
  -------
  cat_cols: list
        kategorik değişken listesi
  num_cols: list
        numerik değişken listesi
  cat_but_car: list
        kategorik görünümlü kordinal değişken listesi



  Notes
  ------
  cat_cols + num_cols + cat_but_car = toplam değişken sayısı
  num_but_cat, cat_cols'un içerisinde.
  """

# Tipi "category", "object", "bool", "str" olanları seç
  cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool", "str"]]

# int, float olup aynı zamanda 10'dan küçük olanları yakala
  num_but_col = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]

# tipleri "category", "object" numerik olarak 20'den büyük olanları getir
  cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]


  cat_cols = cat_cols + num_but_col

  # eğer cat_buttan da bişeyler gelirse ikisinin farkını alıp özelleştir
  cat_cols = [col for col in cat_cols if col not in cat_but_car]

  num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
  num_cols= [col for col in num_cols if col not in cat_cols]

  print(f"Observations: {dataframe.shape[0]}")
  print(f"Variables: {dataframe.shape[1]}")
  print(f"cat_cols: {len(cat_cols)}")
  print(f"num_cols: {len(num_cols)}")
  print(f"cat_but_car: {len(cat_but_car)}")
  print(f"num_but_cat: {len(num_but_col)}")


  return cat_cols, num_cols, cat_but_car

cat_cols,num_cols,cat_but_car = grab_col_names(df)



# kategorik değişkenler
def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
          "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##############################################")

cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df,col)


# numerik değişkenler
def num_summary(dataframe, numeric_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    print(dataframe[numeric_col].describe(quantiles).T)

    if plot:
        dataframe[numeric_col].hist()
        plt.xlabel(numeric_col)
        plt.title(numeric_col)
        plt.show(block=True)

num_summary(df, "age", plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)



# BONUS: Amaç bool tipteki değişkenleri bulup int'e çevirmek ve
# cat_summary fonksiyonunu görsel olarak daha iyi kullanmak

df = sns.load_dataset("titanic")
for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype("int")
cat_cols,num_cols,cat_but_car = grab_col_names(df)



# kategorik değişkenler
def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
          "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##############################################")

cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df,col)











##########################
# Hedef Değişken Analizi
#########################

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option("display.width", 500)
df = pd.read_csv("titanic.csv")
df.head()

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype("int")


def grab_col_names(dataframe, cat_th=10, car_th=20):
  """

  Parameters
  ----------
  dataframe: dataframe
        değişken isimleri alınmak istenen dataframedir
  cat_th: int,float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
  car_th: int,float
        kategorik fakat kordinal değişkenler için sınıf eşik değeri

  Returns
  -------
  cat_cols: list
        kategorik değişken listesi
  num_cols: list
        numerik değişken listesi
  cat_but_car: list
        kategorik görünümlü kordinal değişken listesi



  Notes
  ------
  cat_cols + num_cols + cat_but_car = toplam değişken sayısı
  num_but_cat, cat_cols'un içerisinde.
  """

# Tipi "category", "object", "bool", "str" olanları seç
  cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool", "str"]]

# int, float olup aynı zamanda 10'dan küçük olanları yakala
  num_but_col = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]

# tipleri "category", "object" numerik olarak 20'den büyük olanları getir
  cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]


  cat_cols = cat_cols + num_but_col

  # eğer cat_buttan da bişeyler gelirse ikisinin farkını alıp özelleştir
  cat_cols = [col for col in cat_cols if col not in cat_but_car]

  num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
  num_cols= [col for col in num_cols if col not in cat_cols]

  print(f"Observations: {dataframe.shape[0]}")
  print(f"Variables: {dataframe.shape[1]}")
  print(f"cat_cols: {len(cat_cols)}")
  print(f"num_cols: {len(num_cols)}")
  print(f"cat_but_car: {len(cat_but_car)}")
  print(f"num_but_cat: {len(num_but_col)}")


  return cat_cols, num_cols, cat_but_car

cat_cols,num_cols,cat_but_car = grab_col_names(df)


def target_summary_with_cat(dataframe, target,categorical_col):
    print(pd.DataFrame({"TARGET_MEAN":  dataframe.groupby(categorical_col)[target].mean()}))


target_summary_with_cat(df, "survived","sex")

for col in cat_cols:
    target_summary_with_cat(df,"survived",col)




#################################################
# Hedef Değişkenin Sayısal Değişkenlerle Analizi
#################################################

def target_summary_with_num(dataframe, target,numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

target_summary_with_num(df, "survived","age")
for col in num_cols:
    target_summary_with_num(df,"survived",col)
















#######################
# Korelasyon Analizi
######################
# Korelasyon: değişkenlerin birbirleriyle ilişkisini ifade eden bir ölçümdür.
# -1 ve +1 arasında değerler alır.
# Bir değişkenin değeri artarsa diğerinin de artar.
# Genelde yüksek korelayonlu dizinlerin çalışmalarda bulunmamasını isteriz. Çünkü zaten aynı şeyi ifade ediyor


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option("display.width", 500)

df = pd.read_csv("datasets/breast_cancer.csv")
df = df.iloc[:, 1:-1] # veri setindeki problemli değişkenleri dışarda bırakmak için
df.head()

num_cols = [col for col in df.columns if df[col].dtype in ["int","float"]]

corr = df[num_cols].corr() # korelasyonu hesaplamak için corr() kullanılır


sns.set(rc={'figure.figsize': (12,12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()




##############################################
# Yüksek Korelasyonlu Değişkenlerin Silinmesi
##############################################

cor_matrix = df.corr(numeric_only=True).abs() # mutlak değerden geçirilerek pozitif hale getirdik

upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))


drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.90)]
df.drop(drop_list, axis=1)


def high_correlated_cols(dataframe, plot=False, corr_th = 0.90):
    corr = dataframe.corr(numeric_only=True)
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15,15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

high_correlated_cols(df)
drop_list = high_correlated_cols(df, plot=True)
df.drop(drop_list, axis=1)
high_correlated_cols(df.drop(drop_list, axis = 1), plot=True)
