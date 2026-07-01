

##################################################
# Pandas Alıştırmalar
##################################################

import numpy as np
import seaborn as sns
import pandas as pd

from python_eda import col

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#########################################
# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
#########################################

df= sns.load_dataset("titanic")
df.head()

#########################################
# Görev 2: Yukarıda tanımlanan Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
#########################################

df["sex"].value_counts()


#########################################
# Görev 3: Her bir sütuna ait unique değerlerin sayısını bulunuz.
#########################################

df.nunique()

#########################################
# Görev 4: pclass değişkeninin unique değerleri bulunuz.
#########################################
# unique(): değerleri gösterir    nunique(): değerlerin sayısını gösterir

df["pclass"].unique()

#########################################
# Görev 5:  pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
#########################################

df[["pclass","parch"]].nunique()

#########################################
# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz. Tekrar tipini kontrol ediniz.
#########################################

df["embarked"].dtypes

df["embarked"] = df["embarked"].astype("category")

df["embarked"].dtypes

#########################################
# Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.
#########################################

a = df[df["embarked"] == "C"]
a

#########################################
# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
#########################################

a = df[df["embarked"] != "S"]
a

#########################################
# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
#########################################

df[(df["sex"] == "female") & (df["age"] < 30 )]

#########################################
# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
#########################################

df[(df["fare"] > 500 ) | (df["age"] > 70 )]

#########################################
# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
#########################################

df.isnull().sum()


#########################################
# Görev 12: who değişkenini dataframe'den düşürün.
#########################################

df.drop("who", axis=1).head()

#########################################
# Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
#########################################

mod= df["deck"].mode()[0]

df["deck"].fillna(mod)

#########################################
# Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurun.
#########################################

medyan = df["age"].median()
df["age"].fillna(medyan)

#########################################
# Görev 15: survived değişkeninin Pclass ve Cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
#########################################

df.groupby(["pclass","sex"])["survived"].agg(["sum","count","mean"])

#########################################
# Görev 16:  30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazınız.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)
#########################################

def data_change(dataframe):
    df["new_age"] = np.where(dataframe["age"] < 30 ,0,1)
    print(df["new_age"])
data_change(df)

df.head()





#########################################
# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
#########################################

df = sns.load_dataset("tips")
df.head()

#########################################
# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill  değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################

df.groupby("time")["total_bill"].agg(["sum","min","max","mean"])

#########################################
# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################

df.groupby("day")["total_bill"].agg(["sum","min","max","mean"])

#########################################
# Görev 20:Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
#########################################

koşul = (df["time"] == "Lunch") & (df["sex"] == "Female")


df[koşul].groupby("day")[["total_bill","tip"]].agg(["sum","min","max","mean"])

#########################################
# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?
#########################################

df.head()

siparis= ((df["size"] < 3) & (df["total_bill"] > 10))
siparis.mean()


#########################################
# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturun. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
#########################################

df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]
df["total_bill_tip_sum"]
df.head()

#########################################
# Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
#########################################

a = df.sort_values("total_bill_tip_sum", ascending=False)
top_30 = a.head(30)
top_30





































#############################################
# Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama
#############################################

#############################################
# İş Problemi
#############################################
# Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak seviye tabanlı (level based) yeni müşteri tanımları (persona)
# oluşturmak ve bu yeni müşteri tanımlarına göre segmentler oluşturup bu segmentlere göre yeni gelebilecek müşterilerin şirkete
# ortalama ne kadar kazandırabileceğini tahmin etmek istemektedir.

# Örneğin: Türkiye’den IOS kullanıcısı olan 25 yaşındaki bir erkek kullanıcının ortalama ne kadar kazandırabileceği belirlenmek isteniyor.


#############################################
# Veri Seti Hikayesi
#############################################
# Persona.csv veri seti uluslararası bir oyun şirketinin sattığı ürünlerin fiyatlarını ve bu ürünleri satın alan kullanıcıların bazı
# demografik bilgilerini barındırmaktadır. Veri seti her satış işleminde oluşan kayıtlardan meydana gelmektedir. Bunun anlamı tablo
# tekilleştirilmemiştir. Diğer bir ifade ile belirli demografik özelliklere sahip bir kullanıcı birden fazla alışveriş yapmış olabilir.

# Price: Müşterinin harcama tutarı
# Source: Müşterinin bağlandığı cihaz türü
# Sex: Müşterinin cinsiyeti
# Country: Müşterinin ülkesi
# Age: Müşterinin yaşı

################# Uygulama Öncesi #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

################# Uygulama Sonrası #####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C


#############################################
# PROJE GÖREVLERİ
#############################################

#############################################
# GÖREV 1: Aşağıdaki soruları yanıtlayınız.
#############################################
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)





# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
df = pd.read_csv("datasets/persona.csv")
df.info()
df. head()

# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?
# Frekans: kaç kere tekrar ettiğini gösterir

df["SOURCE"].nunique()
df["SOURCE"].value_counts()


# Soru 3: Kaç unique PRICE vardır?

df["PRICE"].nunique()

# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?

df["PRICE"].value_counts()

# Soru 5: Hangi ülkeden kaçar tane satış olmuş?

df["COUNTRY"].value_counts()

# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?

df.groupby("COUNTRY")["PRICE"].agg(["sum"])


# Soru 7: SOURCE türlerine göre göre satış sayıları nedir?

df["SOURCE"].value_counts()

# Soru 8: Ülkelere göre PRICE ortalamaları nedir?

df.groupby("COUNTRY")["PRICE"].mean()

# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?

df.groupby("SOURCE")["PRICE"].mean()

# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?

df.groupby(["SOURCE","COUNTRY"])["PRICE"].mean()



#############################################
# GÖREV 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
#############################################

df.groupby(["SOURCE","COUNTRY","SEX","AGE"])["PRICE"].mean()

#############################################
# GÖREV 3: Çıktıyı PRICE'a göre sıralayınız.
#############################################
# Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE'a uygulayınız.
# Çıktıyı agg_df olarak kaydediniz.

agg_df = df.groupby(["SOURCE","COUNTRY","SEX","AGE"])["PRICE"].mean().sort_values(ascending=False)
agg_df = df.groupby(["SOURCE","COUNTRY","SEX","AGE"])["PRICE"].mean().reset_index().sort_values(by=["PRICE"], ascending=False)

agg_df.head()

#############################################
# GÖREV 4: Indekste yer alan isimleri değişken ismine çeviriniz.
#############################################
# Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir.
# Bu isimleri değişken isimlerine çeviriniz.
# İpucu: reset_index()
# agg_df.reset_index(inplace=True)

agg_df.reset_index()
agg_df.reset_index(inplace=True)
agg_df.head()

#############################################
# GÖREV 5: AGE değişkenini kategorik değişkene çeviriniz ve agg_df'e ekleyiniz.
#############################################
# Age sayısal değişkenini kategorik değişkene çeviriniz.
# Aralıkları ikna edici olacağını düşündüğünüz şekilde oluşturunuz.
# Örneğin: '0_18', '19_23', '24_30', '31_40', '41_70'

agg_df['AGE'] = agg_df['AGE'].astype('category')

agg_df["NEW_AGE"] = pd.cut(agg_df['AGE'],bins=[0,18,19,23,24,30,31,40,41,70])
agg_df.head()

#############################################
# GÖREV 6: Yeni level based müşterileri tanımlayınız ve veri setine değişken olarak ekleyiniz.
#############################################
# customers_level_based adında bir değişken tanımlayınız ve veri setine bu değişkeni ekleyiniz.
# Dikkat!
# list comp ile customers_level_based değerleri oluşturulduktan sonra bu değerlerin tekilleştirilmesi gerekmektedir.
# Örneğin birden fazla şu ifadeden olabilir: USA_ANDROID_MALE_0_18
# Bunları groupby'a alıp price ortalamalarını almak gerekmektedir.

agg_df["customers_level_based"] = [f"{row[0]}_{row[1]}_{row[2]}_{row[3]}".upper()
                                   for row in agg_df[["SOURCE","COUNTRY","SEX","AGE"]].values]
agg_df.head()

agg_df=agg_df[["customers_level_based","PRICE"]]
agg_df.head()

agg_df = agg_df.groupby(["customers_level_based", "COUNTRY", "SOURCE", "SEX", "NEW_AGE"])["PRICE"].mean().reset_index()

# Sonucu görmek için de doğrudan değişkenin adını yazmalısın:
print(agg_df.head())
#############################################
# GÖREV 7: Yeni müşterileri (USA_ANDROID_MALE_0_18) segmentlere ayırınız.
#############################################
# PRICE'a göre segmentlere ayırınız,
# segmentleri "SEGMENT" isimlendirmesi ile agg_df'e ekleyiniz,
# segmentleri betimleyiniz,

segment_isimleri = ['D', 'C', 'B', 'A']
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], q=4, labels=segment_isimleri)
agg_df.head()

agg_df.groupby("SEGMENT")["PRICE"].agg(["sum","min","max","mean"])




#############################################
# GÖREV 8: Yeni gelen müşterileri sınıflandırınız ne kadar gelir getirebileceğini tahmin ediniz.
#############################################
# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
agg_df.head(50)
hedef_musteri = agg_df[agg_df["customers_level_based"] == "ANDROID_TUR_FEMALE_33"]

print(hedef_musteri[["customers_level_based", "PRICE", "SEGMENT"]])



# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente ve ortalama ne kadar gelir kazandırması beklenir?

hedef_musteri = agg_df[agg_df["customers_level_based"] == "IOS_FRA_FEMALE_35"]

print(hedef_musteri[["customers_level_based", "PRICE", "SEGMENT"]])