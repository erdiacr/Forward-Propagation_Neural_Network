#Tahmin yapabilen bir sinir ağı kurmanın son parçası,
# her şeyi bir araya getirmektir.
# Öyleyse, compute_weighted_sum ve node_activation işlevlerini
# ağdaki her düğüme uygulayan ve verileri çıktı katmanına sonuna kadar uygulayan ve
# çıktı katmanındaki her düğüm için bir tahmin çıkaran bir fonksiyon oluşturalım.


#Bunu başaracağımız yol aşağıdaki prosedürdür:

#1. İlk gizli katmana giriş olarak giriş katmanıyla başlayın.
#2. Geçerli katmanın düğümlerindeki ağırlıklı toplamı hesaplayın.
#3. Geçerli katmanın düğümlerinin çıktısını hesaplayın.
#4. Geçerli katmanın çıktısını bir sonraki katmana girilecek şekilde ayarlayın.
#5. Ağdaki bir sonraki katmana gidin.
#6. Çıktı katmanının çıktısını hesaplayana kadar 2-4. Adımları tekrarlayın.


import numpy as np
import Initialize_NeuralNetwork_Function as INF

def forward_propagate(network, inputs):
    layer_inputs = list(inputs)  # ilk gizli katmana girdi olarak giriş katmanıyla başlayın

    for layer in network:

        layer_data = network[layer]

        layer_outputs = []
        for layer_node in layer_data:
            node_data = layer_data[layer_node]

            # Her bir düğümün ağırlıklı toplamını ve çıktısını aynı anda hesaplayın
            node_output = INF.node_activation(INF.compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias']))
            layer_outputs.append(np.around(node_output[0], decimals=4))

        if layer != 'output':
            print('{}. Gizli katmandaki düğümlerin çıktıları: {}'.format(layer.split('_')[1], layer_outputs))

        layer_inputs = layer_outputs  # bu katmanın çıktısını sonraki katmana girdi olacak şekilde ayarla

    network_predictions = layer_outputs
    return network_predictions


