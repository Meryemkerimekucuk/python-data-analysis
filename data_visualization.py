#######################
# Veri Görselleştirme
#######################

##############################
# Matplotlib: Veri görselleştirmenin atasıdır. Birçok veri görselleştirme kütüphanesinde katkısı vardır.
##############################

# Eğer elimizde kategorik değişken varsa : sütun grafik. countplot, bar
# Sayısal değişken varsa: histogram, boxplot. Bu 2 grafik sayısal değişkenlerin
# hangi aralıklarda hangi frekanslarda gözlemler var bunu gösteririr
# Hem de veri seti içinde nasıl dağılımlar ve birbirine karşı ne kadar aykırıdır
# onu gösterir


#####################################
# Kategorik Değişken Görselleştirme
#####################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

df= sns.load_dataset("titanic")
df.head()


# value_count()= kategorik değişkenlerde aklımıza gelmesi gereken ilk fonksiyondur.

df['sex'].value_counts().plot(kind='bar')
plt.show()


###################################
# Sayısal Değişken Görselleştirme
###################################
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

df= sns.load_dataset("titanic")
df.head()

plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])
plt.show()



#############################
# Matplotlib' in Özellikleri
#############################
# Yapısı itibariyle katmanlı şekilde görselleştirme imkanı sağlar


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

#########
# Plot
#########

x= np.array([1,3,5,7,9])
y= np.array([2,4,6,8,10])

plt.plot(x,y)
plt.show()

plt.plot(x,y, "o")
plt.show()


#####################
# Marker: İşaretleme
#####################

x= np.array([1,3,5,7,9])
y= np.array([2,4,6,8,10])

plt.plot(x,y, marker="*")
plt.show()

#######################
# Line: Çizgi Özelliği
#######################

a= np.array([2,5,8,20,34,12,45,56])
plt.plot(y, linestyle="dashed")
plt.show()

plt.plot(a, linestyle="dashed", color="purple")
plt.show()



##################################
# Multiple Lines: Çoklu Çizgiler
##################################

x= np.array([1,3,5,7,12,14,17,43])
y= np.array([3,4,6,10,15,19,20,34])

plt.plot(x)
plt.plot(y)
plt.show()



######################
# Labels: Etiketler
#####################

x= np.array([1,3,5,7,12,14,17,43])
y= np.array([3,4,6,10,15,19,20,34])
plt.plot(x,y)
plt.show()

plt.title("Ana Başlık")
plt.show()

plt.xlabel("x ekseni")
plt.ylabel("y ekseni")
plt.show()

plt.grid() # ızgara
plt.show()


###############################################
# Subplots: Birden fazla görselin gösterilmesi
###############################################

#plot1
x=np.array([100,200,300,400,500,600])
y=np.array([150,250,350,450,550,650])

plt.title("1")
plt.subplot(1,2,1)
plt.plot(x,y)
plt.show()

#plot2
x=np.array([120,240,360,450,590,670])
y=np.array([450,250,300,120,540,650])

plt.title("2")
plt.subplot(1,2,2)
plt.plot(x,y)
plt.show()






##############################
# Seaborn: Matlotlibe kıyasla daha az çaba harcanarak veri görselleştirme yapılır.
##############################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset("tips")
df.head()

#######################
# Kategorik Değişken
#######################
df["sex"].value_counts()
sns.countplot(x= df["sex"], data=df)
plt.show()


#####################
# Sayısal Değişken
#####################

sns.boxplot(x = df["total_bill"])
plt.show()


