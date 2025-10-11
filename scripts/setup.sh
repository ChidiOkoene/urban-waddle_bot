#!/bin/bash

# Urban Waddle Bot - Setup Script
# This script sets up the trading bot environment and dependencies

set -e

echo "🚀 Setting up Urban Waddle Bot..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION is installed"
    else
        print_error "Python 3.8+ is required but not installed"
        exit 1
    fi
}

# Check if pip is installed
check_pip() {
    print_status "Checking pip installation..."
    if command -v pip3 &> /dev/null; then
        print_success "pip3 is installed"
    else
        print_error "pip3 is required but not installed"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    print_success "Dependencies installed"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p data logs models backups config monitoring/grafana/dashboards monitoring/grafana/datasources nginx/ssl
    print_success "Directories created"
}

# Copy environment template
setup_environment() {
    print_status "Setting up environment configuration..."
    if [ ! -f ".env" ]; then
        cp env.example .env
        print_success "Environment template copied to .env"
        print_warning "Please edit .env file with your actual configuration"
    else
        print_warning ".env file already exists"
    fi
}

# Setup configuration files
setup_config() {
    print_status "Setting up configuration files..."
    if [ ! -f "config/config.yaml" ]; then
        print_warning "config/config.yaml not found. Please create it from the template."
    else
        print_success "Configuration files ready"
    fi
}

# Setup monitoring
setup_monitoring() {
    print_status "Setting up monitoring configuration..."
    
    # Create Prometheus configuration
    cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'trading-bot'
    static_configs:
      - targets: ['trading-bot:8501']
    metrics_path: '/metrics'
    scrape_interval: 5s
EOF

    # Create Grafana datasource
    cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

    print_success "Monitoring configuration created"
}

# Setup Nginx configuration
setup_nginx() {
    print_status "Setting up Nginx configuration..."
    
    cat > nginx/nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    upstream trading_bot {
        server trading-bot:8501;
    }

    server {
        listen 80;
        server_name localhost;

        location / {
            proxy_pass http://trading_bot;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }

        location /metrics {
            proxy_pass http://trading_bot/metrics;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }
    }
}
EOF

    print_success "Nginx configuration created"
}

# Setup database
setup_database() {
    print_status "Setting up database..."
    # Database will be created automatically when the bot starts
    print_success "Database setup ready"
}

# Run tests
run_tests() {
    print_status "Running tests..."
    if [ -f "tests/test_adapters.py" ]; then
        python -m pytest tests/ -v
        print_success "Tests completed"
    else
        print_warning "Test files not found, skipping tests"
    fi
}

# Create systemd service (Linux only)
create_systemd_service() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_status "Creating systemd service..."
        
        cat > urban-waddle-bot.service << EOF
[Unit]
Description=Urban Waddle Trading Bot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

        print_success "Systemd service file created"
        print_warning "To install: sudo cp urban-waddle-bot.service /etc/systemd/system/"
        print_warning "To enable: sudo systemctl enable urban-waddle-bot"
        print_warning "To start: sudo systemctl start urban-waddle-bot"
    fi
}

# Main setup function
main() {
    echo "=========================================="
    echo "Urban Waddle Bot Setup"
    echo "=========================================="
    
    check_python
    check_pip
    create_venv
    activate_venv
    install_dependencies
    create_directories
    setup_environment
    setup_config
    setup_monitoring
    setup_nginx
    setup_database
    
    if [ "$1" = "--with-tests" ]; then
        run_tests
    fi
    
    create_systemd_service
    
    echo "=========================================="
    print_success "Setup completed successfully!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Edit .env file with your configuration"
    echo "2. Edit config/config.yaml with your settings"
    echo "3. Run: ./scripts/start_bot.sh"
    echo ""
    echo "For Docker deployment:"
    echo "1. Run: docker-compose up -d"
    echo ""
    echo "For monitoring:"
    echo "1. Grafana: http://localhost:3000 (admin/admin)"
    echo "2. Prometheus: http://localhost:9090"
    echo ""
}

# Run main function
main "$@"
