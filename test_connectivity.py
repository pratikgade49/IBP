#!/usr/bin/env python3
"""
Connectivity test script to diagnose IBP notification issues
"""
import requests
import urllib3
import logging
from configparser import ConfigParser
import sys

# Disable SSL warnings for testing
urllib3.disable_warnings()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_server_connectivity():
    """Test if the server is accessible from outside"""
    
    # Load config
    config = ConfigParser()
    config.read("server.cfg")
    
    auth_cfg = config["AUTHCONFIG"]
    USER_TOKEN = auth_cfg.get("user_token")
    
    # Test URLs
    test_urls = [
        "https://ec2-107-23-151-16.compute-1.amazonaws.com/ibp/demand/ExternalForecastNotification?RequestID=12345",
        "https://localhost:8001/ibp/demand/ExternalForecastNotification?RequestID=12345",
        "https://127.0.0.1:8001/ibp/demand/ExternalForecastNotification?RequestID=12345"
    ]
    
    headers = {
        "Authorization": USER_TOKEN,
        "User-Agent": "IBP-Connectivity-Test/1.0"
    }
    
    for url in test_urls:
        logger.info(f"Testing connectivity to: {url}")
        try:
            response = requests.get(url, headers=headers, verify=False, timeout=10)
            logger.info(f"✅ SUCCESS - Status: {response.status_code}")
            logger.info(f"Response: {response.text[:200]}...")
        except requests.exceptions.ConnectTimeout:
            logger.error(f"❌ TIMEOUT - Server not reachable at {url}")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"❌ CONNECTION ERROR - {str(e)}")
        except Exception as e:
            logger.error(f"❌ ERROR - {str(e)}")
        
        print("-" * 50)

def check_docker_network():
    """Check Docker network configuration"""
    import subprocess
    
    logger.info("Checking Docker network configuration...")
    
    try:
        # Check if container is running
        result = subprocess.run(['docker', 'ps', '--filter', 'name=ibp-forecast'], 
                              capture_output=True, text=True)
        logger.info(f"Docker ps output:\n{result.stdout}")
        
        # Check port mapping
        result = subprocess.run(['docker', 'port', 'ibp-forecast-1'], 
                              capture_output=True, text=True)
        logger.info(f"Port mapping:\n{result.stdout}")
        
    except Exception as e:
        logger.error(f"Error checking Docker: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting IBP connectivity diagnostics...")
    
    check_docker_network()
    print("=" * 60)
    test_server_connectivity()
    
    logger.info("Diagnostics completed.")