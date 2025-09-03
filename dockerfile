FROM python:3.10-slim

WORKDIR /app

# Create logs directory
RUN mkdir -p /app/logs

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

EXPOSE 8001

CMD ["python", "Server.py"]