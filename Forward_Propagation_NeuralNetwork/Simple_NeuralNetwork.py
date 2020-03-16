#İki girdisi olan,
# iki düğümü olan bir gizli katmanı ve
# bir düğümü olan bir çıkış katmanı olan bir sinir ağı.

#Ağdaki ağırlıkları ve önyargıları rastgele başlatalım.
# Gizli katmandaki her düğüm için ve
# çıkış katmanındaki her düğüm için bir tane olmak üzere
# 6 ağırlık ve 3 bias'a sahibiz.
import numpy as np

weights=np.around(np.random.uniform(size=6),decimals=2) # Ağırlıkların oluşturulması
biases=np.around(np.random.uniform(size=3),decimals=2) # Bias'ın oluşturulması
print("Ağılıklar:",weights)
print("Bias:",biases)

#Şimdi ağ için tanımlanan ağırlıklara ve
# sapmalara sahip olduğumuza göre,
# belirli bir girdi olan x_1 ve x_2 için çıktıyı hesaplayalım.

x_1 = 0.5 # girdi 1
x_2 = 0.85 # girdi 2

print('x1: {} ve x2: {}'.format(x_1, x_2))

#Gizli katmanın ilk düğümünde,
# z_1,1 girişlerinin ağırlıklı toplamını hesaplayarak başlayalım.

z_11 = x_1 * weights[0] + x_2 * weights[1] + biases[0]

print('Gizli katmandaki ilk düğümdeki girdilerin ağırlıklı toplamı z_11: {}'.format(z_11))

#Ardından, gizli katmanın ikinci düğümünde,
# girişlerin ağırlıklı toplamını (z_1,2) hesaplayalım. Değeri z_12 olarak atayalım.

z_12=x_1*weights[2] + x_2*weights[3]+biases[1]
print('Gizli katmandaki ikinci düğümün girdilerinin ağırlıklı toplamı z_12: {}'.format(np.around(z_12, decimals=4)))

#Sonra, sigmoid aktivasyon fonksiyonunu varsayalım,
# ilk katmanın (a_ 1,1) aktivasyonunu gizli katmanda hesaplayalım.

a_11 = 1.0 / (1.0 + np.exp(-z_11))
print('Gizli katmadanki ilk düğümün aktivasyonu a_11: {}'.format(np.around(a_11, decimals=4)))

#Ayrıca gizli katmandaki ikinci düğüm olan a_1,2 'nin aktivasyonunu da hesaplayalım.

a_12=1.0/(1.0+np.exp(-z_12))
print('Gizli katmandaki ikinci düğümün aktivasyonu a_12: {}'.format(np.around(a_12, decimals=4)))

#Şimdi bu aktivasyonlar,
# çıkış katmanının girişleri olarak işlev görecektir.
# Şimdi, bu girdilerin ağırlıklı toplamını
# çıktı katmanındaki düğüme hesaplayalım.

z_2=a_11*weights[4]+a_12*weights[5]+biases[2]
print('Çıkış katmanındaki düğümdeki girdilerin ağırlıklı toplamı z_2: {}'.format(np.around(z_2, decimals=4)))

#Son olarak,
# ağın çıktısını çıktı katmanındaki düğümün aktivasyonu olarak hesaplayalım.
# Değeri a_2'ye atayın.

a_2=1.0/(1.0+np.exp(-z_2))
print('x1 = 0.5 ve x2 = 0.85 için ağın çıktısı a_2: {}'.format(np.around(a_2, decimals=4)))