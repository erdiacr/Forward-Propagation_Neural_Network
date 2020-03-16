#Tahmin yapmanın otomatik bir yolunu kodlamak için ağımızı genelleştirelim.
# Genel bir ağ n girdisi alacak,
# birçok gizli katmana sahip olacak,
# her gizli katman m düğümüne ve
# bir çıkış katmanına sahip olacaktır.

# Ağı birçok gizli katmana sahip olacak şekilde kodlayacağız.
# Benzer şekilde, her ne kadar ağ bir düğümlü bir çıktı katmanı gösterse de,
# ağı çıktı katmanında birden fazla düğüme sahip olacak şekilde kodlayacağız.

#Ağın yapısını tanımlayarak başlayalım.

n = 2 # Girdi sayısı
num_hidden_layers = 2 # Gizli katman sayısı
m = [2, 2] # Her gizli katmandaki düğüm sayısı
num_nodes_output = 1 # Çıktı katmanındaki düğüm sayısı

#Şimdi ağın yapısını tanımladığımıza göre,
# ağdaki ağırlıkları ve önyargıları rastgele sayılarla başlatalım.
# Ağırlıkları ve önyargıları rastgele sayılara başlatabilmek için
# Numpy kütüphanesini içe aktarmamız gerekecek

import numpy as np

num_nodes_previous = n # Önceki katmandaki düğüm sayısı
network = {} # initialize network an an empty dictionary

# Her katmanda döngü yapar ve her bir düğümle ilişkili ağırlıkları ve sapmaları rastgele başlatır
# çıkış katmanını dahil etmek için gizli katmanların sayısına nasıl 1 eklediğimizi fark edin

for layer in range(num_hidden_layers + 1):

    #Katmanın adını belirleme
    if layer == num_hidden_layers:
        layer_name = 'output'
        num_nodes = num_nodes_output

    else:
        layer_name = 'layer_{}'.format(layer + 1)
        num_nodes = m[layer]


    # Geçerli katmandaki her bir düğümle ilişkili ağırlıkları ve biası başlat

    network[layer_name] = {}
    for node in range(num_nodes):
        node_name = 'node_{}'.format(node + 1)
        network[layer_name][node_name] = {
            'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
            'bias': np.around(np.random.uniform(size=1), decimals=2),
        }

    num_nodes_previous = num_nodes

print(network) # Ağı yazdır

#Şimdi yukarıdaki kodla,
# her katmandaki herhangi bir sayıda gizli katman ve
# düğüm sayısının herhangi bir ağına ilişkin ağırlıkları ve sapmaları başlatabiliriz.
# Ancak bu kodu bir fonksiyona koyalım,
# böylece bir nöral ağ kurmak istediğimizde tüm bu kodu tekrar tekrar yürütebiliriz.
