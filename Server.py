import os
import ssl
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from threading import Thread
from configparser import ConfigParser
from ForecastRequestProcessor import process_forecast_request
import logging
import socket
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Change working directory to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load configuration
config = ConfigParser()
config.read("server.cfg")

server_cfg = config["SERVERCONFIG"]
auth_cfg = config["AUTHCONFIG"]

SERVER_ADDRESS = server_cfg.get("server_address", "0.0.0.0")
PORT = server_cfg.getint("port", 8001)
CERT_FILE = server_cfg.get("cert_file", "cert.pem")
KEY_FILE = server_cfg.get("key_file", "privkey.pem")

USER_TOKEN = auth_cfg.get("user_token")

class IBPForecastHandler(SimpleHTTPRequestHandler):
    """
    IBP External Forecast Server Handler
    Handles HTTPS requests from IBP for external forecast notifications
    """

    def do_GET(self):
        start_time = time.time()
        auth_header = self.headers.get("Authorization")
        
        logger.info(f"=== INCOMING REQUEST ===")
        logger.info(f"Path: {self.path}")
        logger.info(f"Method: GET")
        logger.info(f"Client IP: {self.client_address[0]}")
        logger.info(f"User-Agent: {self.headers.get('User-Agent', 'Unknown')}")
        logger.debug(f"Authorization header: {auth_header}")
        logger.debug(f"All headers: {dict(self.headers)}")

        if auth_header is None:
            self._send_unauthorized("No auth header received")
            logger.warning("Request rejected: No authorization header")
            return

        if auth_header != USER_TOKEN:
            self._send_unauthorized("Invalid credentials")
            logger.warning(f"Request rejected: Invalid authorization header")
            return

        # Authenticated request
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        self.wfile.write(b"<html><body>")
        self.wfile.write(b"<h1>IBP External Forecast Server</h1>")

        path = urlparse(self.path).path

        if path == "/ibp/demand/ExternalForecastNotification":
            query = parse_qs(urlparse(self.path).query)
            if "RequestID" in query:
                try:
                    ext_req_id = int(query["RequestID"][0])
                    logger.info(f"Processing external forecast notification for RequestID: {ext_req_id}")
                    
                    self.wfile.write(
                        f"<p>External forecast notification received for RequestID: {ext_req_id}</p>".encode("utf-8")
                    )
                    self.wfile.write(b"<p>Processing started in background...</p>")
                    
                    # Start processing in a separate thread
                    thread = Thread(target=process_forecast_request, args=(ext_req_id,))
                    thread.daemon = True
                    thread.start()
                    
                    logger.info(f"Started background processing for RequestID: {ext_req_id}")
                    
                except ValueError as e:
                    error_msg = f"Invalid RequestID parameter: {query['RequestID'][0]}"
                    logger.error(error_msg)
                    self.wfile.write(f"<p>Error: {error_msg}</p>".encode("utf-8"))
                except Exception as e:
                    error_msg = f"Error processing request: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    self.wfile.write(f"<p>Error: {error_msg}</p>".encode("utf-8"))
            else:
                error_msg = "Missing RequestID parameter"
                logger.warning(error_msg)
                self.wfile.write(f"<p>Error: {error_msg}</p>".encode("utf-8"))
        else:
            logger.warning(f"Invalid request path: {path}")
            self.wfile.write(f"<p>Invalid request path: {path}</p>".encode("utf-8"))
            self.wfile.write(b"<p>Expected: /ibp/demand/ExternalForecastNotification</p>")

        self.wfile.write(b"</body></html>")
        
        end_time = time.time()
        logger.info(f"Request completed in {end_time - start_time:.3f} seconds")
        logger.info("=== REQUEST END ===")

    def do_POST(self):
        """Handle POST requests (in case IBP sends POST instead of GET)"""
        start_time = time.time()
        auth_header = self.headers.get("Authorization")
        
        logger.info(f"=== INCOMING POST REQUEST ===")
        logger.info(f"Path: {self.path}")
        logger.info(f"Method: POST")
        logger.info(f"Client IP: {self.client_address[0]}")
        logger.info(f"User-Agent: {self.headers.get('User-Agent', 'Unknown')}")
        logger.debug(f"Authorization header: {auth_header}")
        logger.debug(f"All headers: {dict(self.headers)}")
        
        # Read POST body if present
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length > 0:
            post_data = self.rfile.read(content_length)
            logger.debug(f"POST body: {post_data.decode('utf-8', errors='ignore')}")

        if auth_header is None:
            self._send_unauthorized("No auth header received")
            logger.warning("POST request rejected: No authorization header")
            return

        if auth_header != USER_TOKEN:
            self._send_unauthorized("Invalid credentials")
            logger.warning(f"POST request rejected: Invalid authorization header")
            return

        # Process the same way as GET
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        self.wfile.write(b"<html><body>")
        self.wfile.write(b"<h1>IBP External Forecast Server</h1>")

        path = urlparse(self.path).path

        if path == "/ibp/demand/ExternalForecastNotification":
            query = parse_qs(urlparse(self.path).query)
            if "RequestID" in query:
                try:
                    ext_req_id = int(query["RequestID"][0])
                    logger.info(f"Processing external forecast notification (POST) for RequestID: {ext_req_id}")
                    
                    self.wfile.write(
                        f"<p>External forecast notification received for RequestID: {ext_req_id}</p>".encode("utf-8")
                    )
                    self.wfile.write(b"<p>Processing started in background...</p>")
                    
                    # Start processing in a separate thread
                    thread = Thread(target=process_forecast_request, args=(ext_req_id,))
                    thread.daemon = True
                    thread.start()
                    
                    logger.info(f"Started background processing for RequestID: {ext_req_id}")
                    
                except ValueError as e:
                    error_msg = f"Invalid RequestID parameter: {query['RequestID'][0]}"
                    logger.error(error_msg)
                    self.wfile.write(f"<p>Error: {error_msg}</p>".encode("utf-8"))
                except Exception as e:
                    error_msg = f"Error processing request: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    self.wfile.write(f"<p>Error: {error_msg}</p>".encode("utf-8"))
            else:
                error_msg = "Missing RequestID parameter"
                logger.warning(error_msg)
                self.wfile.write(f"<p>Error: {error_msg}</p>".encode("utf-8"))
        else:
            logger.warning(f"Invalid POST request path: {path}")
            self.wfile.write(f"<p>Invalid request path: {path}</p>".encode("utf-8"))
            self.wfile.write(b"<p>Expected: /ibp/demand/ExternalForecastNotification</p>")

        self.wfile.write(b"</body></html>")
        
        end_time = time.time()
        logger.info(f"POST request completed in {end_time - start_time:.3f} seconds")
        logger.info("=== POST REQUEST END ===")

    def _send_unauthorized(self, message):
        self.send_response(401)
        self.send_header("WWW-Authenticate", "Basic realm='IBP External Forecast Server'")
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(f"<html><body><h1>401 Unauthorized</h1><p>{message}</p></body></html>".encode("utf-8"))

    def log_message(self, format, *args):
        # Override to use our logger instead of stderr
        logger.info(f"{self.address_string()} - {format % args}")

def run_server():
    try:
        # Create HTTPS server
        httpd = HTTPServer((SERVER_ADDRESS, PORT), IBPForecastHandler)
        
        # Configure SSL context
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=CERT_FILE, keyfile=KEY_FILE)
        
        # Wrap socket with SSL
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
        
        logger.info(f"IBP External Forecast Server started at https://{SERVER_ADDRESS}:{PORT}")
        logger.info("Waiting for external forecast notification requests from IBP...")
        logger.info(f"Endpoint: https://ec2-107-23-151-16.compute-1.amazonaws.com/ibp/demand/ExternalForecastNotification")
        
        # Log network diagnostics
        logger.info(f"Server listening on {SERVER_ADDRESS}:{PORT}")
        logger.info(f"SSL certificate: {CERT_FILE}")
        logger.info(f"SSL private key: {KEY_FILE}")
        
        # Test if port is accessible
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.bind((SERVER_ADDRESS, PORT))
            test_socket.close()
            logger.info(f"Port {PORT} is available and bound successfully")
        except Exception as e:
            logger.error(f"Port binding test failed: {str(e)}")

        httpd.serve_forever()
        
    except FileNotFoundError as e:
        logger.error(f"SSL certificate files not found: {str(e)}")
        logger.error("Please ensure cert.pem and privkey.pem are in the current directory")
    except ssl.SSLError as e:
        logger.error(f"SSL configuration error: {str(e)}")
    except Exception as e:
        logger.error(f"Server startup error: {str(e)}", exc_info=True)
    finally:
        logger.info("Server stopped.")

if __name__ == "__main__":
    run_server()