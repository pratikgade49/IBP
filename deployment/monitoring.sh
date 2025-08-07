#!/bin/bash

# Monitoring Script for IBP External Forecasting
# Run this script to check system health

echo "=== IBP External Forecasting - System Health Check ==="
echo "Timestamp: $(date)"
echo ""

# Check service status
echo "1. Service Status:"
sudo systemctl is-active ibp-forecasting
sudo systemctl is-enabled ibp-forecasting
echo ""

# Check process
echo "2. Process Information:"
ps aux | grep -E "(Server.py|python.*Server)" | grep -v grep
echo ""

# Check ports
echo "3. Port Status:"
sudo netstat -tlnp | grep -E ":(80|443|8443)"
echo ""

# Check SSL certificate
echo "4. SSL Certificate Status:"
if [ -f "/etc/letsencrypt/live/*/fullchain.pem" ]; then
    sudo certbot certificates
else
    echo "No Let's Encrypt certificates found"
fi
echo ""

# Check disk space
echo "5. Disk Usage:"
df -h /
echo ""

# Check memory usage
echo "6. Memory Usage:"
free -h
echo ""

# Check recent logs
echo "7. Recent Service Logs (last 10 lines):"
sudo journalctl -u ibp-forecasting -n 10 --no-pager
echo ""

# Check Nginx status
echo "8. Nginx Status:"
sudo systemctl is-active nginx
sudo nginx -t 2>&1
echo ""

# Check application logs
echo "9. Application Logs (last 5 lines):"
if [ -f "/opt/ibp-forecasting/forecast_processor.log" ]; then
    tail -5 /opt/ibp-forecasting/forecast_processor.log
else
    echo "No application log file found"
fi
echo ""

# Test endpoint
echo "10. Health Check:"
curl -s -o /dev/null -w "HTTP Status: %{http_code}\nResponse Time: %{time_total}s\n" http://localhost/health 2>/dev/null || echo "Health check failed"
echo ""

echo "=== Health Check Complete ==="