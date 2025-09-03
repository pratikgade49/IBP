# from http.server import HTTPServer, SimpleHTTPRequestHandler
# import ssl
# from urllib.parse import urlparse, parse_qs
# from threading import Thread
# from ForecastRequestProcessor import process_forecast_request
# import os
# from configparser import ConfigParser

# os.chdir(os.path.dirname(os.path.abspath(__file__)))

# # Read config file
# config_object = ConfigParser()
# config_object.read("server.cfg")
# server_cfg = config_object["SERVERCONFIG"]

# # Certificate and key path
# KEY_FILE = server_cfg["key_file"]
# CERT_FILE = server_cfg["cert_file"]

# # Server address and port
# SERVER_ADDRESS = server_cfg["server_address"]
# PORT = int(server_cfg["port"])

# USER_TOKEN = config_object["AUTHCONFIG"]["user_token"]

# class SampleForecastServer(SimpleHTTPRequestHandler):
#     """
#     Sample external forecast server implementation for development. Do not use it in production systems!

#     Args:
#         SimpleHTTPRequestHandler (class): Python built in simple HTTP server class
#     """

#     def do_GET(self):

#         if self.headers["Authorization"] is None:
#             self.send_response(401)
#             self.send_header("WWW-Authenticate", "Basic realm=\'Test\'")
#             self.send_header("Content-type", "text/html")
#             self.end_headers()
#             self.wfile.write(bytes("No auth header received", "utf-8"))
#             print("No auth header received")

#         elif self.headers["Authorization"] == USER_TOKEN:
#             self.send_response(200)
#             self.send_header("Content-type", "text/html")
#             self.end_headers()

#             self.wfile.write(bytes("<body>", "utf-8"))
#             self.wfile.write(bytes("<p>Hello IBP.</p>", "utf-8"))

#             path = urlparse(self.path).path

#             if path == "/ibp/demand/ExternalForecastNotification":
#                 ext_req_id = int(
#                     parse_qs(urlparse(self.path).query)["RequestID"][0])
#                 self.wfile.write(bytes(f"<p>External forecast notification received for id: \
#                      {ext_req_id}. Processing triggered!</p>", "utf-8"))

#                 thread = Thread(target=process_forecast_request,
#                                 args=(ext_req_id,))
#                 thread.start()

#             else:
#                 self.wfile.write(
#                     bytes(f"<p>Invalid request: {self.path}</p>", "utf-8"))

#             self.wfile.write(bytes("</body></html>", "utf-8"))
#         else:
#             self.send_response(401)
#             self.send_header("WWW-Authenticate", "Basic realm=\'Test\'")
#             self.send_header("Content-type", "text/html")
#             self.end_headers()
#             self.wfile.write(bytes(self.headers["Authorization"], "utf-8"))
#             self.wfile.write(bytes("Unauthenticated", "utf-8"))
#             print("Unauthenticated")


# if __name__ == "__main__":
#     webServer = HTTPServer((SERVER_ADDRESS, PORT), SampleForecastServer)

#     context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
#     context.load_cert_chain(certfile=CERT_FILE,
#                             keyfile=KEY_FILE)
#     webServer.socket = context.wrap_socket(webServer.socket, server_side=True)

#     print("Server started. Waiting for external notification requests.")
#     try:
#         webServer.serve_forever()
#     except (KeyboardInterrupt, SystemExit):
#         pass
#     webServer.server_close()
#     print("Server stopped.")

import os
import ssl
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from threading import Thread
from configparser import ConfigParser
from ForecastRequestProcessor import process_forecast_request

# Change working directory to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load configuration
config = ConfigParser()
config.read("server.cfg")

server_cfg = config["SERVERCONFIG"]
auth_cfg = config["AUTHCONFIG"]

SERVER_ADDRESS = server_cfg.get("server_address", "0.0.0.0")
PORT = server_cfg.getint("port", 8000)
KEY_FILE = server_cfg.get("KEY_FILE", "newprivkey.pem")
CERT_FILE = server_cfg.get("CERT_FILE", "cert.pem")

USER_TOKEN = auth_cfg.get("user_token")

class ForecastHTTPRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        auth_header = self.headers.get("Authorization")

        if auth_header is None:
            self._send_unauthorized("No auth header received")
            print("No auth header received")
            return

        if auth_header != USER_TOKEN:
            self._send_unauthorized("Unauthenticated")
            print(f"Unauthenticated access attempt with header: {auth_header}")
            return

        # Authenticated
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        self.wfile.write(b"<html><body>")
        self.wfile.write(b"<p>Hello IBP.</p>")

        path = urlparse(self.path).path

        if path == "/ibp/demand/ExternalForecastNotification":
            query = parse_qs(urlparse(self.path).query)
            if "RequestID" in query:
                try:
                    ext_req_id = int(query["RequestID"][0])
                    self.wfile.write(
                        f"<p>External forecast notification received for id: {ext_req_id}. Processing triggered!</p>".encode("utf-8")
                    )
                    # Start processing in a separate thread
                    thread = Thread(target=process_forecast_request, args=(ext_req_id,))
                    thread.daemon = True
                    thread.start()
                except ValueError:
                    self.wfile.write(b"<p>Invalid RequestID parameter.</p>")
            else:
                self.wfile.write(b"<p>Missing RequestID parameter.</p>")
        else:
            self.wfile.write(f"<p>Invalid request path: {path}</p>".encode("utf-8"))

        self.wfile.write(b"</body></html>")

    def _send_unauthorized(self, message):
        self.send_response(401)
        self.send_header("WWW-Authenticate", "Basic realm='IBP Server'")
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(message.encode("utf-8"))

def run_server():
    httpd = HTTPServer((SERVER_ADDRESS, PORT), ForecastHTTPRequestHandler)

    # Setup SSL context
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=CERT_FILE, keyfile=KEY_FILE)
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

    print(f"Server started at https://{SERVER_ADDRESS}:{PORT}")
    print("Waiting for external notification requests...")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopping...")
    finally:
        httpd.server_close()
        print("Server stopped.")

if __name__ == "__main__":
    run_server()
