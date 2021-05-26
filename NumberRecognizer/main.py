from NumberNeuralNet import NumberNeuralNet
from SimpleHttpServer import SimpleHttpServer

class main:
    def __init__(self):
        nnn = NumberNeuralNet(0.05, 500, [40, 20])
        self.server = SimpleHttpServer("localhost", 3030, nnn)

if __name__ == "__main__":
    m = main()
    