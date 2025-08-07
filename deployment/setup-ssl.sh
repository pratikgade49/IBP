#!/bin/bash

# SSL Setup Script for IBP External Forecasting
# Run this after install.sh and configuring your domain

set -e

echo "=== SSL Certificate Setup ==="

# Check if domain is provided
if [ -z "$1" ]; then
    echo "Usage: ./setup-ssl.sh your-domain.com"
    echo "Example: ./setup-ssl.sh forecast.mycompany.com"
    exit 1
fi

DOMAIN=$1
EMAIL="admin@$DOMAIN"

echo "Setting up SSL for domain: $DOMAIN"

# Create Nginx configuration
sudo tee /etc/nginx/sites-available/ibp-forecasting > /dev/null <<EOF
server {
    listen 80;
    server_name $DOMAIN;
    
    # Redirect HTTP to HTTPS
    return 301 https://\$server_name\$request_uri;
}

server {
    listen 443 ssl http2;
    server_name $DOMAIN;
    
    # SSL Configuration (will be updated by certbot)
    ssl_certificate /etc/letsencrypt/live/$DOMAIN/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$DOMAIN/privkey.pem;
    
    # SSL Security Settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    
    # Proxy to Python application
    location / {
        proxy_pass https://127.0.0.1:8443;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Handle SSL verification
        proxy_ssl_verify off;
        proxy_ssl_session_reuse on;
        
        # Timeouts for long-running forecasts
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
EOF

# Enable site
sudo ln -sf /etc/nginx/sites-available/ibp-forecasting /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test Nginx configuration
sudo nginx -t

# Obtain SSL certificate
echo "Obtaining SSL certificate from Let's Encrypt..."
sudo certbot --nginx -d $DOMAIN --non-interactive --agree-tos --email $EMAIL

# Reload Nginx
sudo systemctl reload nginx

# Generate self-signed certificate for Python application
echo "Generating self-signed certificate for Python application..."
sudo mkdir -p /opt/ibp-forecasting/certs
cd /opt/ibp-forecasting/certs

# Generate private key
sudo openssl genrsa -out server.key 2048

# Generate certificate signing request
sudo openssl req -new -key server.key -out server.csr -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

# Generate self-signed certificate
sudo openssl x509 -req -days 365 -in server.csr -signkey server.key -out server.crt

# Set permissions
sudo chown $USER:$USER /opt/ibp-forecasting/certs/*
sudo chmod 600 /opt/ibp-forecasting/certs/server.key
sudo chmod 644 /opt/ibp-forecasting/certs/server.crt

echo "=== SSL Setup Complete! ==="
echo "Your service will be available at: https://$DOMAIN"
echo "Internal Python app uses self-signed cert at: https://127.0.0.1:8443"
echo ""
echo "Next steps:"
echo "1. Update server.cfg with the correct paths"
echo "2. Start the service: sudo systemctl start ibp-forecasting"
echo "3. Check status: sudo systemctl status ibp-forecasting"