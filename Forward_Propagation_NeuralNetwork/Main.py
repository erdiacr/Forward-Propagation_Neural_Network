import Forward_Propagate_Function as FPF
import Initialize_NeuralNetwork_Function as INF
import numpy as np
from random import seed

n = 10 # Girdi sayısı
num_hidden_layers = 5 # Gizli katman sayısı
num_node_of_hidden_layers = [2, 3, 5, 6, 8] # Her gizli katmandaki düğüm sayısı
num_nodes_output = 3 # Çıktı katmanındaki düğüm sayısı


def input(n): # Girdi fonksiyonu, Girdiler rastgele belirlendi.( Girdi değeri 5)
    np.random.seed(12)
    inputs = np.around(np.random.uniform(size=n), decimals=2)
    return  inputs

inputs=input(n)

print('Girdi Değeri N:',n, 'Gizli katman sayısı:',num_hidden_layers, 'Çıktı Katmanındaki Düğüm Sayısı:', num_nodes_output)
print('Ağın girdi değerleri: {}'.format(inputs))

#### Small_Network'un tahminini hesaplamak için * forward_propagate * fonksiyonunu kullanın

small_network=INF.initialize_network(n,num_hidden_layers,num_node_of_hidden_layers,num_nodes_output)
predictions = FPF.forward_propagate(small_network, inputs)
print('Ağ tarafından verilen girdi için tahmin edilen çıktı: {}'.format(np.around(predictions, decimals=4)))




