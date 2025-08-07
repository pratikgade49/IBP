#!/bin/bash

# AWS EC2 Installation Script for IBP External Forecasting
# Run this script on your EC2 instance after initial setup

set -e

echo "=== IBP External Forecasting - AWS EC2 Installation ==="

# Update system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python 3.9+ and pip
echo "Installing Python and dependencies..."
sudo apt install -y python3 python3-pip python3-venv git nginx certbot python3-certbot-nginx

# Create application directory
echo "Setting up application directory..."
sudo mkdir -p /opt/ibp-forecasting
sudo chown $USER:$USER /opt/ibp-forecasting
cd /opt/ibp-forecasting

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Create systemd service
echo "Creating systemd service..."
sudo tee /etc/systemd/system/ibp-forecasting.service > /dev/null <<EOF
[Unit]
Description=IBP External Forecasting Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/ibp-forecasting
Environment=PATH=/opt/ibp-forecasting/venv/bin
ExecStart=/opt/ibp-forecasting/venv/bin/python Server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable ibp-forecasting
echo "Service created. Will start after SSL setup."

echo "=== Installation completed! ==="
echo "Next steps:"
echo "1. Configure your domain in server.cfg"
echo "2. Run setup-ssl.sh to configure SSL"
echo "3. Start the service with: sudo systemctl start ibp-forecasting"