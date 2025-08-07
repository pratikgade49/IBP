# AWS EC2 Deployment Guide for IBP External Forecasting

## Prerequisites

1. **AWS Account** with EC2 access
2. **Domain name** pointing to your EC2 instance
3. **SAP IBP system** credentials and access

## Step-by-Step Deployment

### 1. Launch EC2 Instance

```bash
# Launch Ubuntu 22.04 LTS instance
# Recommended: t3.medium or larger (2 vCPU, 4GB RAM minimum)
# Storage: 20GB+ SSD

# Instance type recommendations:
# - Development: t3.medium (2 vCPU, 4GB RAM)
# - Production: t3.large or c5.large (2-4 vCPU, 8GB+ RAM)
# - High-volume: c5.xlarge or m5.xlarge (4+ vCPU, 16GB+ RAM)
```

### 2. Configure Security Group

```bash
# Create security group using AWS CLI
aws ec2 create-security-group \
    --group-name ibp-forecasting-sg \
    --description "Security Group for IBP External Forecasting"

# Add rules (or use the provided JSON file)
aws ec2 authorize-security-group-ingress \
    --group-name ibp-forecasting-sg \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-name ibp-forecasting-sg \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-name ibp-forecasting-sg \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0
```

### 3. Connect to Instance and Deploy

```bash
# SSH to your instance
ssh -i your-key.pem ubuntu@your-ec2-public-ip

# Clone your repository or upload files
git clone your-repo-url
# OR
scp -i your-key.pem -r . ubuntu@your-ec2-public-ip:~/ibp-forecasting/

# Navigate to project directory
cd ibp-forecasting

# Run installation script
chmod +x deployment/install.sh
./deployment/install.sh
```

### 4. Configure Domain and SSL

```bash
# Update server.cfg with your SAP credentials
nano server.cfg

# Set up SSL with your domain
chmod +x deployment/setup-ssl.sh
./deployment/setup-ssl.sh your-domain.com
```

### 5. Start the Service

```bash
# Start the forecasting service
sudo systemctl start ibp-forecasting

# Check status
sudo systemctl status ibp-forecasting

# View logs
sudo journalctl -u ibp-forecasting -f
```

### 6. Test the Deployment

```bash
# Test health endpoint
curl https://your-domain.com/health

# Test IBP endpoint (replace with your token)
curl -H "Authorization: Bearer your_token" \
     "https://your-domain.com/ibp/demand/ExternalForecastNotification?RequestID=123"
```

## Configuration Files

### server.cfg
Update the following sections:
- `[SERVICECONFIG]`: Your SAP IBP server URL
- `[AUTHCONFIG]`: SAP credentials and secure token

### Security Considerations

1. **Restrict SSH access** to your IP only
2. **Use strong passwords** and consider key-based authentication
3. **Generate secure tokens** for API access
4. **Enable CloudWatch monitoring**
5. **Set up automated backups**

## Monitoring and Maintenance

### Log Files
- Application logs: `/opt/ibp-forecasting/forecast_processor.log`
- System logs: `sudo journalctl -u ibp-forecasting`
- Nginx logs: `/var/log/nginx/access.log` and `/var/log/nginx/error.log`

### Service Management
```bash
# Start service
sudo systemctl start ibp-forecasting

# Stop service
sudo systemctl stop ibp-forecasting

# Restart service
sudo systemctl restart ibp-forecasting

# View status
sudo systemctl status ibp-forecasting

# Enable auto-start on boot
sudo systemctl enable ibp-forecasting
```

### SSL Certificate Renewal
```bash
# Certificates auto-renew, but you can test renewal:
sudo certbot renew --dry-run

# Force renewal if needed:
sudo certbot renew --force-renewal
```

## Troubleshooting

### Common Issues

1. **Service won't start**
   ```bash
   sudo journalctl -u ibp-forecasting -n 50
   ```

2. **SSL certificate issues**
   ```bash
   sudo certbot certificates
   sudo nginx -t
   ```

3. **Python dependencies**
   ```bash
   cd /opt/ibp-forecasting
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Port conflicts**
   ```bash
   sudo netstat -tlnp | grep :8443
   sudo lsof -i :8443
   ```

### Performance Tuning

For high-volume forecasting:
1. Increase EC2 instance size
2. Add more worker processes
3. Implement load balancing
4. Consider using AWS Application Load Balancer
5. Set up CloudWatch alarms for monitoring

## Backup and Recovery

1. **Regular snapshots** of EBS volumes
2. **Configuration backup** of server.cfg
3. **Log rotation** to prevent disk space issues
4. **Database backup** if using external storage

## Cost Optimization

1. Use **Reserved Instances** for production
2. **Stop instances** during non-business hours if applicable
3. **Monitor CloudWatch** for resource utilization
4. Consider **Spot Instances** for development/testing