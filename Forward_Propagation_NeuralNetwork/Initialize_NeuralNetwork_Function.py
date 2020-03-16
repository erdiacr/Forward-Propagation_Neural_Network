import numpy as np

def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    num_nodes_previous = num_inputs  # Önceki katmandaki düşüm sayısı

    network = {}

    # Her katmanda döngü yapar ve her bir düğümle ilişkili ağırlıkları ve sapmaları rastgele başlatır
    for layer in range(num_hidden_layers + 1):

        if layer == num_hidden_layers:
            layer_name = 'output'  # ağ çıkışındaki son katmanı adlandır
            num_nodes = num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1)  # aksi takdirde katmana bir sayı verir
            num_nodes = num_nodes_hidden[layer]

            # Her düğüm için ağırlıkları ve biasları başlatır
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node + 1)
            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'bias': np.around(np.random.uniform(size=1), decimals=2),
            }

        num_nodes_previous = num_nodes

    return network  # Ağı yazdırır

#### Aşağıdakileri yapan bir ağ oluşturmak için * initialize_network * fonksiyonunu kullanın:
#1. 5 giriş alır
#2. üç gizli katmanı var
#3. birinci katmanda 3, ikinci katmanda 2 düğüm ve üçüncü katmanda 3 düğüm vardır
#4. çıkış katmanında 1 düğümü var

#Ağı ** small_network ** olarak adlandırın.

#small_network=initialize_network(5,3,[3,2,3],1)

#Her bir düğümdeki ağırlıklı toplam,
# girişlerin ve ağırlığın + biasın nokta çarpımı olarak hesaplanır.
# Öyleyse adlı bir fonksiyon oluşturalım.

def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias

#Small_network'e vereceğimiz 5 girdi oluşturalım

#from random import seed
#def input(): # Girdi fonksiyonu, Girdiler rastgele belirlendi.
    #np.random.seed(12)
    #inputs = np.around(np.random.uniform(size=5), decimals=2)
    #return  inputs
#print('Ağın girdi değerleri: {}'.format(input()))

#İlk gizli katmandaki ilk düğümdeki ağırlıklı toplamı hesaplamak için compute_weighted_sum fonksiyonunu kullanın

#node_weights = small_network['layer_1']['node_1']['weights']
#node_bias = small_network['layer_1']['node_1']['bias']

#weighted_sum = compute_weighted_sum(input(), node_weights, node_bias)
#print('Gizli katmandaki ilk düğümün ağırlıklı toplamı: {}'.format(np.around(weighted_sum[0], decimals=4)))

#Her düğümün çıkışının,
# ağırlıklı toplamın doğrusal olmayan bir dönüşümü olduğunu hatırlayın.
# Bu eşleme için aktivasyon fonksiyonlarını kullanıyoruz.
# Burada sigmoid fonksiyonunu aktivasyon fonksiyonu olarak kullanalım.
# Öyleyse ağırlıklı toplamı girdi olarak alan ve
# sigmoid işlevini kullanarak girdinin doğrusal olmayan dönüşümünü döndüren bir fonksiyonu tanımlayalım.

def node_activation(weighted_sum):
    return 1.0 / (1.0 + np.exp(-1 * weighted_sum))

# İlk gizli katmandaki ilk düğümün çıktısını hesaplamak için * node_activation * fonksiyonunu kullanın.

#node_output  = node_activation(compute_weighted_sum(input(), node_weights, node_bias))
#print('Gizli katmandaki ilk düğümün çıktısı{}'.format(np.around(node_output[0], decimals=4)))





