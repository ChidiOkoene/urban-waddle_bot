#!/bin/bash

# Urban Waddle Bot - Database Backup Script
# This script creates automated backups of the trading bot database

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
BACKUP_DIR="backups"
DB_PATH="data/trading_bot.db"
RETENTION_DAYS=7
COMPRESS=true
NOTIFY=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --backup-dir)
            BACKUP_DIR="$2"
            shift 2
            ;;
        --db-path)
            DB_PATH="$2"
            shift 2
            ;;
        --retention-days)
            RETENTION_DAYS="$2"
            shift 2
            ;;
        --no-compress)
            COMPRESS=false
            shift
            ;;
        --no-notify)
            NOTIFY=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --backup-dir DIR         Backup directory (default: backups)"
            echo "  --db-path PATH          Database file path (default: data/trading_bot.db)"
            echo "  --retention-days DAYS   Number of days to keep backups (default: 7)"
            echo "  --no-compress           Disable compression"
            echo "  --no-notify             Disable notifications"
            echo "  --help                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 --retention-days 14 --no-compress"
            echo "  $0 --db-path /path/to/db.db --backup-dir /path/to/backups"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if database exists
check_database() {
    if [ ! -f "$DB_PATH" ]; then
        print_error "Database file $DB_PATH not found"
        exit 1
    fi
}

# Create backup directory
create_backup_dir() {
    if [ ! -d "$BACKUP_DIR" ]; then
        print_status "Creating backup directory: $BACKUP_DIR"
        mkdir -p "$BACKUP_DIR"
        print_success "Backup directory created"
    fi
}

# Create backup filename
create_backup_filename() {
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local db_name=$(basename "$DB_PATH" .db)
    
    if [ "$COMPRESS" = true ]; then
        BACKUP_FILENAME="${db_name}_backup_${timestamp}.db.gz"
    else
        BACKUP_FILENAME="${db_name}_backup_${timestamp}.db"
    fi
    
    BACKUP_PATH="$BACKUP_DIR/$BACKUP_FILENAME"
}

# Create database backup
create_backup() {
    print_status "Creating database backup..."
    print_status "Source: $DB_PATH"
    print_status "Destination: $BACKUP_PATH"
    
    # Check database integrity first
    print_status "Checking database integrity..."
    sqlite3 "$DB_PATH" "PRAGMA integrity_check;" > /dev/null
    
    if [ $? -eq 0 ]; then
        print_success "Database integrity check passed"
    else
        print_error "Database integrity check failed"
        exit 1
    fi
    
    # Create backup
    if [ "$COMPRESS" = true ]; then
        sqlite3 "$DB_PATH" ".backup main '$BACKUP_PATH.tmp'"
        gzip "$BACKUP_PATH.tmp"
        mv "$BACKUP_PATH.tmp.gz" "$BACKUP_PATH"
    else
        sqlite3 "$DB_PATH" ".backup main '$BACKUP_PATH'"
    fi
    
    if [ $? -eq 0 ]; then
        print_success "Backup created successfully"
        
        # Get backup size
        BACKUP_SIZE=$(du -h "$BACKUP_PATH" | cut -f1)
        print_status "Backup size: $BACKUP_SIZE"
    else
        print_error "Backup creation failed"
        exit 1
    fi
}

# Clean old backups
clean_old_backups() {
    print_status "Cleaning old backups (older than $RETENTION_DAYS days)..."
    
    local deleted_count=0
    
    # Find and delete old backup files
    while IFS= read -r -d '' file; do
        print_status "Deleting old backup: $(basename "$file")"
        rm "$file"
        ((deleted_count++))
    done < <(find "$BACKUP_DIR" -name "*_backup_*.db*" -type f -mtime +$RETENTION_DAYS -print0)
    
    if [ $deleted_count -gt 0 ]; then
        print_success "Deleted $deleted_count old backup(s)"
    else
        print_status "No old backups to delete"
    fi
}

# Verify backup
verify_backup() {
    print_status "Verifying backup..."
    
    if [ "$COMPRESS" = true ]; then
        # Test compressed backup
        if gzip -t "$BACKUP_PATH" 2>/dev/null; then
            print_success "Compressed backup verification passed"
        else
            print_error "Compressed backup verification failed"
            exit 1
        fi
    else
        # Test SQLite backup
        if sqlite3 "$BACKUP_PATH" "PRAGMA integrity_check;" > /dev/null 2>&1; then
            print_success "Backup verification passed"
        else
            print_error "Backup verification failed"
            exit 1
        fi
    fi
}

# Send notification
send_notification() {
    if [ "$NOTIFY" = true ]; then
        print_status "Sending notification..."
        
        local message="✅ Database backup completed successfully
📁 Backup: $BACKUP_FILENAME
📊 Size: $BACKUP_SIZE
🕒 Time: $(date)"

        # Try to send via Telegram if configured
        if [ ! -z "$TELEGRAM_BOT_TOKEN" ] && [ ! -z "$TELEGRAM_CHAT_ID" ]; then
            curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
                -d "chat_id=$TELEGRAM_CHAT_ID" \
                -d "text=$message" \
                -d "parse_mode=Markdown" > /dev/null 2>&1
            
            if [ $? -eq 0 ]; then
                print_success "Telegram notification sent"
            else
                print_warning "Failed to send Telegram notification"
            fi
        fi
        
        # Try to send via Discord if configured
        if [ ! -z "$DISCORD_WEBHOOK_URL" ]; then
            curl -s -X POST "$DISCORD_WEBHOOK_URL" \
                -H "Content-Type: application/json" \
                -d "{\"content\": \"$message\"}" > /dev/null 2>&1
            
            if [ $? -eq 0 ]; then
                print_success "Discord notification sent"
            else
                print_warning "Failed to send Discord notification"
            fi
        fi
    fi
}

# Create backup summary
create_summary() {
    local summary_file="$BACKUP_DIR/backup_summary.txt"
    
    cat > "$summary_file" << EOF
Urban Waddle Bot - Database Backup Summary
==========================================

Backup Details:
- Date: $(date)
- Source: $DB_PATH
- Destination: $BACKUP_PATH
- Size: $BACKUP_SIZE
- Compressed: $COMPRESS
- Retention: $RETENTION_DAYS days

Database Statistics:
- Tables: $(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM sqlite_master WHERE type='table';")
- Total Records: $(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM trades;" 2>/dev/null || echo "N/A")

Backup History:
$(ls -la "$BACKUP_DIR"/*_backup_*.db* 2>/dev/null | tail -10 || echo "No previous backups found")
EOF

    print_status "Backup summary created: $summary_file"
}

# Main function
main() {
    echo "=========================================="
    echo "Urban Waddle Bot - Database Backup"
    echo "=========================================="
    
    check_database
    create_backup_dir
    create_backup_filename
    create_backup
    verify_backup
    clean_old_backups
    create_summary
    send_notification
    
    echo "=========================================="
    print_success "Backup completed successfully!"
    echo "=========================================="
    echo ""
    echo "Backup Details:"
    echo "  File: $BACKUP_FILENAME"
    echo "  Size: $BACKUP_SIZE"
    echo "  Location: $BACKUP_PATH"
    echo ""
    echo "To restore from backup:"
    echo "  gunzip $BACKUP_PATH && sqlite3 data/trading_bot.db < ${BACKUP_PATH%.gz}"
    echo ""
}

# Run main function
main "$@"
