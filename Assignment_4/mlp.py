from layer import Layer
import numpy as np

class MLP():

    def __init__(self, nodes_per_layer):

        self.layers = []

        for i in range(len(nodes_per_layer)):
            if (i==0):
                prev = 'start'
            else:
                prev = nodes_per_layer[i-1]

            if (i==len(nodes_per_layer) - 1):
                next = 'end'
            else:
                next = nodes_per_layer[i+1]

            self.layers.append(Layer(nodes_per_layer[i],next_layer_number = next, previous_layer_number = prev))

    def __str__(self):
        layer_str = []

        max_nodes = 0

        for i in self.layers:
            layer = str(i)
            layer_str.append(layer.split('\n'))
            if layer.count('\n') > max_nodes:
                max_nodes = layer.count('\n')

        string = ""
        for i in range(max_nodes):
            for j in layer_str:
                if(len(j) > i+1):
                    string += j[i].center(9)
                else:
                    string += "         "
            string += "\n"

        return string

    def predict(self, point, regression):
        dat = np.transpose(point)
        self.layers[0].set_nodes(dat)
        for i in range(len(self.layers)-1):
            nxt_lyr = self.layers[i].feed_forward_sigmoid()
            self.layers[i+1].set_nodes(nxt_lyr)
        if(regression):
            nxt_lyr = self.layers[-2].feed_forward()
        else:
            nxt_lyr = self.layers[-2].feed_forward_sigmoid()
        self.layers[-1].set_nodes(nxt_lyr)

        #print(self.layers[-1])
        return(self.layers[-1])
