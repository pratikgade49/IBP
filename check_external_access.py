#!/usr/bin/env python3
"""
External connectivity test - simulates IBP calling your server
"""
import requests
import urllib3
import logging
from configparser import ConfigParser
import sys
import time

# Disable SSL warnings
urllib3.disable_warnings()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_external_connectivity():
    """Test external connectivity to the server"""
    
    # Load config
    config = ConfigParser()
    config.read("server.cfg")
    
    auth_cfg = config["AUTHCONFIG"]
    USER_TOKEN = auth_cfg.get("user_token")
    
    # Test the actual endpoint that IBP should be calling
    test_url = "https://ec2-107-23-151-16.compute-1.amazonaws.com/ibp/demand/ExternalForecastNotification?RequestID=12345"
    health_url = "https://ec2-107-23-151-16.compute-1.amazonaws.com/health"
    
    headers = {
        "Authorization": USER_TOKEN,
        "User-Agent": "IBP-External-Test/1.0"
    }
    
    print("=" * 60)
    print("EXTERNAL CONNECTIVITY TEST")
    print("=" * 60)
    
    # Test 1: Health check (no auth required)
    logger.info("Testing health endpoint (no auth)...")
    try:
        response = requests.get(health_url, verify=False, timeout=10)
        if response.status_code == 200:
            logger.info("[OK] Health check successful - Server is reachable!")
            print(f"Response: {response.text}")
        else:
            logger.error(f"[FAIL] Health check failed - Status: {response.status_code}")
    except Exception as e:
        logger.error(f"[FAIL] Health check failed - {str(e)}")
    
    print("-" * 40)
    
    # Test 2: Authenticated endpoint
    logger.info("Testing IBP notification endpoint (with auth)...")
    try:
        response = requests.get(test_url, headers=headers, verify=False, timeout=10)
        if response.status_code == 200:
            logger.info("[OK] IBP endpoint test successful!")
            print(f"Response: {response.text}")
        elif response.status_code == 401:
            logger.warning("[WARN] Authentication failed - Check credentials")
        else:
            logger.error(f"[FAIL] IBP endpoint test failed - Status: {response.status_code}")
    except Exception as e:
        logger.error(f"[FAIL] IBP endpoint test failed - {str(e)}")
    
    print("-" * 40)
    
    # Test 3: Check if it's a Docker networking issue
    logger.info("Testing local Docker connectivity...")
    local_health_url = "https://localhost:8001/health"
    try:
        response = requests.get(local_health_url, verify=False, timeout=5)
        if response.status_code == 200:
            logger.info("[OK] Local Docker connectivity works")
        else:
            logger.error(f"[FAIL] Local Docker test failed - Status: {response.status_code}")
    except Exception as e:
        logger.error(f"[FAIL] Local Docker test failed - {str(e)}")

def check_aws_security_groups():
    """Provide instructions for checking AWS security groups"""
    print("\n" + "=" * 60)
    print("AWS SECURITY GROUP CHECKLIST")
    print("=" * 60)
    print("Please verify the following in your AWS Console:")
    print("1. Go to EC2 → Security Groups")
    print("2. Find the security group attached to your EC2 instance")
    print("3. Check Inbound Rules:")
    print("   - Type: HTTPS")
    print("   - Protocol: TCP")
    print("   - Port Range: 443")
    print("   - Source: 0.0.0.0/0 (or specific IBP IP ranges)")
    print("4. If the rule doesn't exist, add it")
    print("5. Also check if there's a rule for port 8001 if needed")

def check_ibp_configuration():
    """Provide IBP configuration checklist"""
    print("\n" + "=" * 60)
    print("IBP CONFIGURATION CHECKLIST")
    print("=" * 60)
    print("In your IBP system, verify:")
    print("1. External Algorithm URL is set to:")
    print("   https://ec2-107-23-151-16.compute-1.amazonaws.com/ibp/demand/ExternalForecastNotification")
    print("2. Authentication is configured with:")
    print(f"   Username: {config['AUTHCONFIG']['username']}")
    print("   Password: [Check if password is correct]")
    print("3. The algorithm is properly activated")
    print("4. Test the connection from IBP admin panel if available")

if __name__ == "__main__":
    logger.info("Starting external connectivity diagnostics...")
    
    # Load config for IBP checklist
    config = ConfigParser()
    config.read("server.cfg")
    
    test_external_connectivity()
    check_aws_security_groups()
    check_ibp_configuration()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Run this test script to verify connectivity")
    print("2. Check AWS security groups if health check fails")
    print("3. Verify IBP configuration if auth test fails")
    print("4. Check IBP logs for any error messages")