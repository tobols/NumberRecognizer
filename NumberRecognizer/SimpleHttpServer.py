from http.server import BaseHTTPRequestHandler, HTTPServer
from NumberNeuralNet import NumberNeuralNet
import json

class SimpleHttpServer:
    def __init__(self, hostName, serverPort, neural_net):
        def handler(*args):
            SimpleHandler(neural_net, *args)
        print("Server started http://%s:%s" % (hostName, serverPort))
        server = HTTPServer((hostName, serverPort), handler)
        server.serve_forever()



class SimpleHandler(BaseHTTPRequestHandler):
    def __init__(self, neural_net, *args):
        self.neural_net = neural_net
        BaseHTTPRequestHandler.__init__(self, *args)


    def do_OPTIONS(self):           
        self.send_response(200, "ok")       
        self.send_header('Access-Control-Allow-Origin', '*')                
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "*")   
        self.end_headers()


    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(self.response("It's alive!"))


    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data)
        prediction, acc = self.neural_net.predict(data["pixels"])
        response = json.dumps({'prediction':int(prediction[0]), 'accuracy': acc})
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response.encode("utf8"))


    def response(self, message):
        content = f"<html><body><h1>{message}</h1></body></html>"
        return content.encode("utf8")



class main:
    def __init__(self):
        nnn = NumberNeuralNet(0.05, 1, [10, 10], 1)
        self.server = SimpleHttpServer("localhost", 3030, nnn)



if __name__ == '__main__':
    m = main()
