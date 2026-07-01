#Python ile Veri Analizi

#Numpy = Matematiksel İşlemlerde kullanılır
#Pandas = Veri analizi, veri manipülasyonlarında kullanılır
#Veri Görselleştirme: Matplotlib, Seaborn
#Keşifsel Veri Analizi


#############
# Numpy
############
# Matematiksel işlemler için kullanılır.
# Arraylar, çok boyutlu arraylar, ve matrisler üzerinde yüksek performanslı çalışma imkanı sağlar.
# Numpyın listelerden farklılaştığı nokta:
# 1. verimli veri saklama
# 2. yüksek seviyeli işlemlerdir
# Neden numpy= daha az çabayla daha az işlem yapmamızı sağlar. Döngüllerle yapılan uzun işlemler yerine numpylar ile daha kısa bir şekilde yapabiliriz


import numpy as np

# for döngüsü ile
a = [1, 2, 3, 4, 5]
b = [2, 4, 6, 8, 10]

ab = []
for i in range(0, len(a)):
    ab.append(a[i] * b[i])
    print(ab)

#numpy ile
a = np.array([1, 2, 3, 4, 5])
b = np.array([2, 4, 6, 8, 10])
a * b

##########################
# Numpy arrayi oluşturma
##########################

type(np.array([1, 2, 3, 4, 5]))  # 'numpy.ndarray' ndarray demek numpy array demektir

np.zeros(10, dtype=int)  # 10 adet int tipinde array oluştur

np.random.randint(0, 10, size=10)  # 0-10 arası 10 adet rastgele sayı üret

np.random.normal(10, 6, (3, 4))  # ortalaması 10, standart sapması 6 olan 3 satır 4 sütun oluşan array üret

#############################
# Numpy array özellikleri
#############################

# ndim: boyut sayısı
# shape: boyut bilgisi
# size: eleman sayısı
# dtype: array veri tipi

import numpy as np

a = np.random.randint(0, 10, size=5)

a.ndim  #boyut sayısı
a.shape  #boyut bilgisi
a.size  #eleman sayısı
a.dtype  #tipi

######################################
# Yeniden Şekillendirme (reshaping)
######################################
import numpy as np

a = np.random.randint(0, 20, size=16)
a.reshape(4, 4)

#################
# Index Seçimi
#################

import numpy as np

a = np.random.randint(0, 10, size=5)
a[3]

a[0:4]

b = np.random.randint(0, 20, size=(4, 5))  # 2 boyutlu array oluşturma
b

b[0, 0]  # virgülden önce satır, virgülden sonra sütunu ifade eder

b[0, 4] = 100
b

b[1, 3] = 3.20
b

b[1, :]  # 1.satırdaki bütün sütunları seç demek

#################
# Fancy Index
#################

import numpy as np

n = np.arange(0, 30, 3) # 0-30 arasında 3'er 3'er artar array oluştur

catch = [1, 2, 3]
n[catch]

#############################
# Numpyda Koşullu İşlemler
#############################
import numpy as np

v = np.array([1, 2, 3, 4, 5])

#Normal döngü ile
ab = []
for i in v:
    if i < 3:
        ab.append(i)

#Numpy ile
v < 3
v[v < 3]

#################################
# Numpyda Matematiksel İşlemler
################################

# Operatörler ile
v = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

v * 2

v / 5

v ** 2

# Metotlar ile

np.subtract(v, 5)  # çıkarma işlemi
np.add(v, 10)  # toplama işlemi
np.sum(v)  # bütün elemanların toplamı
np.mean(v)  # elemanların ortalaması
np.max(v)  # en büyük eleman
np.min(v)  # en küçük eleman
np.var(v)  # varyans alma

##########################################
# Numpy ile 2 bilinmeyenli denklem çözümü
##########################################

# 5*x0 + x1= 12
# x0 + 3*x1= 10

a = np.array([[5, 1], [1, 3]])
b = np.array([12, 10])

np.linalg.solve(a, b)

############
# Pandas
############
# Veri analitiği ve manipülasyonu için kullanılır


##################
# Pandas Series
##################
# Tek boyutlu ve indeks yapısı barındıran tek boyutlu veri tipidir

# pandas dataframe: çok boyutlu ve indeks bilgisi barındıran veri tipidir

import pandas as pd

s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

s.index
s.dtypes
s.size
s.ndim
s.values
s.head()  # ilk 5 veri
s.tail()  # son 5 veri

######################
# Veri Okuma
######################

import pandas as pd

df = pd.read_csv("datasets/advertising.csv")

df.head()

########################
# Veriye Hızlı Bakış
########################

import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()  # ilk 5 satır
df.tail()  # son 5 satır
df.shape  # boyut bilgisi
df.info()  # detaylı bilgi için
df.columns  # değişken isimlerine erişmek için
df.index  # index bilgisi için
df.describe().T  # T daha okunabilir olması için
df.isnull().any()  # hiç null değeri var mı
df.isnull().sum()  # değişkenlerdeki null değerlerin toplamı
df["sex"].head()
df["sex"].value_counts()  # değişkenden kaçar tane olduğunu göstermek için

############################
# Pandasta Seçim İşlemleri
############################

import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()

# Bu silme işlemleri kalıcı değildir
df.drop(0, axis=0).head()  # 0. indexten satır bilgisini sil
delete_index = [1, 3, 5, 7]
df.drop(delete_index, axis=0).head(10)

# Kalıcı hale getirilmesi isteniyorsa
# 1. yol değişken atanabilir
# 2. yol değişken atanmadan inplace ile kalıcı hale getirilebilir: df.drop(delete_index,axis=0, inplace=True)


############################
# Değişkeni Indexe Çevirme
############################

import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()

df["age"].head()
df.age.head()
df.index = df["age"]
df.drop("age", axis=1).head()
df.drop("age", axis=1, inplace=True)
df.head()

###########################
# Indexi Değişkene Çevirme
###########################

# 1. yol
df["age"] = df.index
df.head()
df.drop("age", axis=1, inplace=True)

# 2. yol
df.reset_index().head()
df = df.reset_index().head()
df.head()

#################################
# Değişkenler Üzerinde İşlemler
#################################

import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)  # Bütün columnsları yan yana göster demek
df = sns.load_dataset("titanic")

df.head()

"age" in df  # Bu değişken bu veri setinde var mı demektir

# !!!!!!!!!!!!!!!!!!
# Bir değişken seçerken sonucu dataframe veya seri olarak alınabilir

type(df["age"].head())  # seri
type(df[["age"]].head())  # dataframe

df[["sex", "age", "alive"]]
col_name = ["sex", "age", "alive"]
df[col_name]

df["age2"] = df["age"] * 2
df["age2"].head()
df.drop("age2", axis=1, inplace=True)
df

# loc dataframelerde seçme işlemi için kullanılan özel yapıdır. label based
df.loc[:, df.columns.str.contains("age")].head()  # age ifadesinde string değerlerde sütunlarda var mı yok mu ara. ilk 5 satırı getir
df.loc[:, ~df.columns.str.contains("age")].head()  # ~işareti ile bunun dışındakileri sil

#####################
# iloc ve loc yapısı
#####################
# iloc: index bilgisi vererek seçim yapmamızı sağlar. index based
# loc: indexlerdeki labellara göre seçim yapar. label based
import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")

df.iloc[0:3]
df.iloc[0, 3]  # 0: satır , 3: sütun

df.loc[0:3]

df.iloc[0:3, 0:3]  # virgülden sonraki 0:3 anlamı: 0dan 3. değişkene kadar git
df.loc[0:3, "age"]  # age değişkenini göster

#############################
# Koşullu Seçim İşlemleri
#############################

import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")

df[df["age"] > 60].head()
df[df["age"] > 60].count()

df[df["age"] > 60]["age"].count()

df.loc[df["age"] > 60, "class"].head()

df.loc[(df["age"] > 60) & (df["sex"] == "male"), ["age", "class"]].head()

######################################################
# Toplulaştırma ve Gruplama (Aggregation & Grouping)
######################################################
# Toplulaştırma: Bir veri yapısının içinde bulunan değerleri toplu bir şekilde temsil etme
# Gruplama: özet istatistikleri veren fonksiyonlardır.

import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")

df["age"].mean()
df.groupby("sex")["age"].mean()  # gruplama.   Cinsiyete göre gruplama ve yaş değişkenine göre ortalamasını al

df.groupby("sex").agg({"age": "mean"})

df.groupby("sex").agg({"age": ["mean", "sum"]}) # agg, "aggregate" (toplulaştırmak / bir araya getirmek) kelimesinin kısaltmasıdır.

##################
# Pivot Table
##################
import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")

df.pivot_table("survived", "sex","embarked")  # survived: kesişim, sex: satır, embarked: sütun ayrıca ön tanımlı değer meandir
df.pivot_table("survived", "sex", "embarked", aggfunc="std")
df.pivot_table("survived", "sex", ["embarked", "class"])

# cut ve qcut sayısal değişkeni kategorik değişkene çevirmek için kullanılır
# cut: sayısal değişkeni hangi kategorik değişkene çevireceğimizi biliyorsak kullanırız
# qcut: sayısal değişkeni tanımıyorum, çeyreklik değerlerine bölünsün istiyorum

df["new_age"] = pd.cut(df["age"], [0, 10, 20, 30, 40, 50])
df.head()

df.pivot_table("survived", "sex", ["new_age", "class"])

###################
# Apply & Lambda
##################

# Apply: satır veya sütunlarda otomatik olarak kod çalıştırmayı sağlar
# Lambda: Kullan at fonksiyon tanımlamadır

import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")

df["age2"] = df["age"] * 2
df["age3"] = df["age"] * 3

# 1.yol
df[["age", "age2", "age3"]].apply(lambda x: x / 10).head()

# 2.yol
df.loc[:, df.columns.str.contains("age")].apply(lambda x: x / 10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x - x.mean()) / x.std()).head()


"""
Eğer amacınız sadece konsolda veriye hızlıca bakmak (Read-only) ise
ve sütun adlarını kesin biliyorsanız 1. yol gayet iyidir.

Ancak bir script yazıyorsanız, veriyi kalıcı olarak değiştirecekseniz
veya sütun isimleri dinamik olarak değişebilecek bir yapıdaysa kesinlikle 
2. yolu (df.loc) alışkanlık haline getirmek en doğrusudur.

"""




################################
# Birleştirme (Join) İşlemleri
################################

"""
concat(), yapıştırma (alt alta veya yan yana) işlemidir.

merge(), ortak bir anahtara (key) göre ilişkilendirme
(veritabanlarındaki SQL JOIN mantığı) işlemidir.
"""

# concat ile
import pandas as pd
import numpy as np

m = np.random.randint(1, 30, (5, 3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
df2 = df1 + 99

pd.concat([df1, df2])

# merge ile

df1 = pd.DataFrame({"Employees": ["John", "Denis", "Mark", "Maria"],
                    "Group": ["Accounting", "Engineering", "Engineering", "HR"]})

df2 = pd.DataFrame({"Employees": ["Mark", "John", "Denis", "Maria"],
                    "Start_Date": [2010, 2009, 20014, 2009]})

df3 = pd.merge(df1, df2)

df4 = pd.DataFrame({"Group": ["Accounting", "Engineering", "HR"],
                    "Manager": ["Caner", "Mustafa", "Berkcan"]})

pd.merge(df3, df4)
