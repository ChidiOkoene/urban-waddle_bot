#!/bin/bash

# Urban Waddle Bot - Start Script
# This script starts the trading bot with proper configuration

set -e

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

# Default values
MODE="paper"
STRATEGY="rsi_macd"
CONFIG_FILE="config/config.yaml"
LOG_LEVEL="INFO"
DASHBOARD_PORT="8501"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --dashboard-port)
            DASHBOARD_PORT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode MODE              Trading mode: paper, live, backtest (default: paper)"
            echo "  --strategy STRATEGY      Strategy to use (default: rsi_macd)"
            echo "  --config FILE            Configuration file (default: config/config.yaml)"
            echo "  --log-level LEVEL        Log level: DEBUG, INFO, WARNING, ERROR (default: INFO)"
            echo "  --dashboard-port PORT    Dashboard port (default: 8501)"
            echo "  --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --mode paper --strategy rsi_macd"
            echo "  $0 --mode live --strategy bollinger_mean_reversion --log-level DEBUG"
            echo "  $0 --mode backtest --strategy ema_crossover"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if virtual environment exists
check_venv() {
    if [ ! -d "venv" ]; then
        print_error "Virtual environment not found. Please run ./scripts/setup.sh first"
        exit 1
    fi
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Check if .env file exists
check_env() {
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating from template..."
        if [ -f "env.example" ]; then
            cp env.example .env
            print_warning "Please edit .env file with your configuration before running live trading"
        else
            print_error "env.example template not found"
            exit 1
        fi
    fi
}

# Check if config file exists
check_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Configuration file $CONFIG_FILE not found"
        exit 1
    fi
}

# Check if required directories exist
check_directories() {
    print_status "Checking directories..."
    mkdir -p data logs models backups
    print_success "Directories ready"
}

# Validate configuration
validate_config() {
    print_status "Validating configuration..."
    
    # Check if Python can import the config
    python -c "
import sys
sys.path.append('src')
try:
    from src.config.config_loader import ConfigLoader
    config_loader = ConfigLoader('$CONFIG_FILE')
    config = config_loader.load_config()
    print('Configuration validation passed')
except Exception as e:
    print(f'Configuration validation failed: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_success "Configuration validation passed"
    else
        print_error "Configuration validation failed"
        exit 1
    fi
}

# Check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    python -c "
import sys
required_packages = [
    'pandas', 'numpy', 'ccxt', 'streamlit', 'plotly',
    'scikit-learn', 'xgboost', 'lightgbm', 'pydantic',
    'pyyaml', 'aiosqlite', 'python-telegram-bot'
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f'Missing packages: {missing_packages}')
    sys.exit(1)
else:
    print('All required packages are installed')
"
    
    if [ $? -eq 0 ]; then
        print_success "Dependencies check passed"
    else
        print_error "Missing required packages. Please run ./scripts/setup.sh"
        exit 1
    fi
}

# Start the bot
start_bot() {
    print_status "Starting Urban Waddle Bot..."
    print_status "Mode: $MODE"
    print_status "Strategy: $STRATEGY"
    print_status "Config: $CONFIG_FILE"
    print_status "Log Level: $LOG_LEVEL"
    print_status "Dashboard Port: $DASHBOARD_PORT"
    
    # Set environment variables
    export BOT_MODE="$MODE"
    export ACTIVE_STRATEGY="$STRATEGY"
    export LOG_LEVEL="$LOG_LEVEL"
    export DASHBOARD_PORT="$DASHBOARD_PORT"
    export CONFIG_FILE="$CONFIG_FILE"
    
    # Start the bot
    python main.py \
        --mode "$MODE" \
        --strategy "$STRATEGY" \
        --config "$CONFIG_FILE" \
        --log-level "$LOG_LEVEL" \
        --dashboard-port "$DASHBOARD_PORT"
}

# Handle signals
handle_signals() {
    print_status "Setting up signal handlers..."
    
    cleanup() {
        print_warning "Received shutdown signal. Stopping bot gracefully..."
        # Send SIGTERM to the bot process
        if [ ! -z "$BOT_PID" ]; then
            kill -TERM "$BOT_PID" 2>/dev/null || true
            wait "$BOT_PID" 2>/dev/null || true
        fi
        print_success "Bot stopped gracefully"
        exit 0
    }
    
    trap cleanup SIGINT SIGTERM
}

# Main function
main() {
    echo "=========================================="
    echo "Urban Waddle Bot Startup"
    echo "=========================================="
    
    check_venv
    activate_venv
    check_env
    check_config
    check_directories
    validate_config
    check_dependencies
    handle_signals
    
    echo "=========================================="
    print_success "All checks passed. Starting bot..."
    echo "=========================================="
    
    start_bot
}

# Run main function
main "$@"
