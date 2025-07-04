# Example base image and dependencies
FROM python:3.10-slim
# ai-ml Docker image
FROM python:3.10-slim

# Install common utilities
RUN pip install --no-cache-dir common==v1.0.0

# Copy code
WORKDIR /app
COPY . .

# Install Python dependencies (you’ll add requirements.txt soon)
# RUN pip install -r requirements.txt

CMD ["bash"]
EOF
