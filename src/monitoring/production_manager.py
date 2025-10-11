"""
Production Monitoring and Health Check System

This module provides comprehensive monitoring, health checks,
and production-ready features for the Urban Waddle Bot.
"""

import asyncio
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.core.data_models import Balance, Position, Trade
from src.database.db_manager import DatabaseManager


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None


class SystemMonitor:
    """System resource monitoring."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        return psutil.cpu_percent(interval=1)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information."""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percentage': memory.percent
        }
    
    def get_disk_usage(self) -> Dict[str, Any]:
        """Get disk usage information."""
        disk = psutil.disk_usage('/')
        return {
            'total': disk.total,
            'used': disk.used,
            'free': disk.free,
            'percentage': (disk.used / disk.total) * 100
        }
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        network = psutil.net_io_counters()
        return {
            'bytes_sent': network.bytes_sent,
            'bytes_recv': network.bytes_recv,
            'packets_sent': network.packets_sent,
            'packets_recv': network.packets_recv
        }
    
    def get_process_info(self) -> Dict[str, Any]:
        """Get current process information."""
        process = psutil.Process()
        return {
            'pid': process.pid,
            'name': process.name(),
            'cpu_percent': process.cpu_percent(),
            'memory_percent': process.memory_percent(),
            'memory_info': process.memory_info()._asdict(),
            'num_threads': process.num_threads(),
            'create_time': process.create_time()
        }


class TradingBotMonitor:
    """Trading bot specific monitoring."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self.start_time = datetime.now()
        self.last_heartbeat = datetime.now()
        self.trade_count = 0
        self.error_count = 0
    
    async def get_bot_status(self) -> Dict[str, Any]:
        """Get overall bot status."""
        uptime = datetime.now() - self.start_time
        
        return {
            'uptime_seconds': uptime.total_seconds(),
            'uptime_human': str(uptime),
            'start_time': self.start_time.isoformat(),
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'trade_count': self.trade_count,
            'error_count': self.error_count,
            'status': 'running' if self.is_healthy() else 'unhealthy'
        }
    
    async def get_trading_metrics(self) -> Dict[str, Any]:
        """Get trading performance metrics."""
        try:
            # Get recent trades
            trades = await self.db_manager.get_trades(limit=100)
            
            if not trades:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'total_pnl': 0.0,
                    'avg_trade_duration': 0.0
                }
            
            # Calculate metrics
            total_trades = len(trades)
            winning_trades = sum(1 for trade in trades if trade.pnl and trade.pnl > 0)
            losing_trades = sum(1 for trade in trades if trade.pnl and trade.pnl < 0)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            total_pnl = sum(trade.pnl or 0 for trade in trades)
            
            # Calculate average trade duration
            durations = []
            for trade in trades:
                if trade.exit_time and trade.entry_time:
                    duration = (trade.exit_time - trade.entry_time).total_seconds()
                    durations.append(duration)
            
            avg_trade_duration = sum(durations) / len(durations) if durations else 0.0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_trade_duration': avg_trade_duration
            }
            
        except Exception as e:
            self.logger.error(f"Error getting trading metrics: {e}")
            return {}
    
    async def get_position_metrics(self) -> Dict[str, Any]:
        """Get position metrics."""
        try:
            positions = await self.db_manager.get_positions()
            
            if not positions:
                return {
                    'total_positions': 0,
                    'total_exposure': 0.0,
                    'unrealized_pnl': 0.0,
                    'largest_position': 0.0
                }
            
            total_positions = len(positions)
            total_exposure = sum(pos.size * pos.current_price for pos in positions)
            unrealized_pnl = sum(pos.unrealized_pnl or 0 for pos in positions)
            largest_position = max((pos.size * pos.current_price for pos in positions), default=0.0)
            
            return {
                'total_positions': total_positions,
                'total_exposure': total_exposure,
                'unrealized_pnl': unrealized_pnl,
                'largest_position': largest_position
            }
            
        except Exception as e:
            self.logger.error(f"Error getting position metrics: {e}")
            return {}
    
    def update_heartbeat(self):
        """Update heartbeat timestamp."""
        self.last_heartbeat = datetime.now()
    
    def increment_trade_count(self):
        """Increment trade count."""
        self.trade_count += 1
    
    def increment_error_count(self):
        """Increment error count."""
        self.error_count += 1
    
    def is_healthy(self) -> bool:
        """Check if bot is healthy."""
        # Bot is healthy if heartbeat is within last 5 minutes
        return (datetime.now() - self.last_heartbeat).total_seconds() < 300


class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self, db_manager: DatabaseManager, bot_monitor: TradingBotMonitor):
        self.db_manager = db_manager
        self.bot_monitor = bot_monitor
        self.system_monitor = SystemMonitor()
        self.logger = logging.getLogger(__name__)
    
    async def check_database_health(self) -> HealthCheck:
        """Check database connectivity and performance."""
        try:
            # Test database connection
            start_time = time.time()
            await self.db_manager.get_trades(limit=1)
            response_time = time.time() - start_time
            
            if response_time > 5.0:
                return HealthCheck(
                    name="database",
                    status=HealthStatus.WARNING,
                    message=f"Database response time is slow: {response_time:.2f}s",
                    timestamp=datetime.now(),
                    details={'response_time': response_time}
                )
            elif response_time > 10.0:
                return HealthCheck(
                    name="database",
                    status=HealthStatus.CRITICAL,
                    message=f"Database response time is too slow: {response_time:.2f}s",
                    timestamp=datetime.now(),
                    details={'response_time': response_time}
                )
            else:
                return HealthCheck(
                    name="database",
                    status=HealthStatus.HEALTHY,
                    message="Database is responding normally",
                    timestamp=datetime.now(),
                    details={'response_time': response_time}
                )
                
        except Exception as e:
            return HealthCheck(
                name="database",
                status=HealthStatus.CRITICAL,
                message=f"Database connection failed: {str(e)}",
                timestamp=datetime.now()
            )
    
    async def check_system_resources(self) -> HealthCheck:
        """Check system resource usage."""
        try:
            cpu_usage = self.system_monitor.get_cpu_usage()
            memory_usage = self.system_monitor.get_memory_usage()
            disk_usage = self.system_monitor.get_disk_usage()
            
            # Check CPU usage
            if cpu_usage > 90:
                return HealthCheck(
                    name="system_resources",
                    status=HealthStatus.CRITICAL,
                    message=f"CPU usage is critically high: {cpu_usage:.1f}%",
                    timestamp=datetime.now(),
                    details={
                        'cpu_usage': cpu_usage,
                        'memory_usage': memory_usage['percentage'],
                        'disk_usage': disk_usage['percentage']
                    }
                )
            elif cpu_usage > 80:
                return HealthCheck(
                    name="system_resources",
                    status=HealthStatus.WARNING,
                    message=f"CPU usage is high: {cpu_usage:.1f}%",
                    timestamp=datetime.now(),
                    details={
                        'cpu_usage': cpu_usage,
                        'memory_usage': memory_usage['percentage'],
                        'disk_usage': disk_usage['percentage']
                    }
                )
            
            # Check memory usage
            if memory_usage['percentage'] > 95:
                return HealthCheck(
                    name="system_resources",
                    status=HealthStatus.CRITICAL,
                    message=f"Memory usage is critically high: {memory_usage['percentage']:.1f}%",
                    timestamp=datetime.now(),
                    details={
                        'cpu_usage': cpu_usage,
                        'memory_usage': memory_usage['percentage'],
                        'disk_usage': disk_usage['percentage']
                    }
                )
            elif memory_usage['percentage'] > 85:
                return HealthCheck(
                    name="system_resources",
                    status=HealthStatus.WARNING,
                    message=f"Memory usage is high: {memory_usage['percentage']:.1f}%",
                    timestamp=datetime.now(),
                    details={
                        'cpu_usage': cpu_usage,
                        'memory_usage': memory_usage['percentage'],
                        'disk_usage': disk_usage['percentage']
                    }
                )
            
            # Check disk usage
            if disk_usage['percentage'] > 95:
                return HealthCheck(
                    name="system_resources",
                    status=HealthStatus.CRITICAL,
                    message=f"Disk usage is critically high: {disk_usage['percentage']:.1f}%",
                    timestamp=datetime.now(),
                    details={
                        'cpu_usage': cpu_usage,
                        'memory_usage': memory_usage['percentage'],
                        'disk_usage': disk_usage['percentage']
                    }
                )
            elif disk_usage['percentage'] > 85:
                return HealthCheck(
                    name="system_resources",
                    status=HealthStatus.WARNING,
                    message=f"Disk usage is high: {disk_usage['percentage']:.1f}%",
                    timestamp=datetime.now(),
                    details={
                        'cpu_usage': cpu_usage,
                        'memory_usage': memory_usage['percentage'],
                        'disk_usage': disk_usage['percentage']
                    }
                )
            
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.HEALTHY,
                message="System resources are within normal limits",
                timestamp=datetime.now(),
                details={
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory_usage['percentage'],
                    'disk_usage': disk_usage['percentage']
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check system resources: {str(e)}",
                timestamp=datetime.now()
            )
    
    async def check_bot_health(self) -> HealthCheck:
        """Check trading bot health."""
        try:
            if not self.bot_monitor.is_healthy():
                return HealthCheck(
                    name="trading_bot",
                    status=HealthStatus.CRITICAL,
                    message="Trading bot heartbeat is stale",
                    timestamp=datetime.now(),
                    details={
                        'last_heartbeat': self.bot_monitor.last_heartbeat.isoformat(),
                        'uptime': (datetime.now() - self.bot_monitor.start_time).total_seconds()
                    }
                )
            
            # Check error rate
            if self.bot_monitor.error_count > 10:
                return HealthCheck(
                    name="trading_bot",
                    status=HealthStatus.WARNING,
                    message=f"High error count: {self.bot_monitor.error_count}",
                    timestamp=datetime.now(),
                    details={
                        'error_count': self.bot_monitor.error_count,
                        'trade_count': self.bot_monitor.trade_count
                    }
                )
            
            return HealthCheck(
                name="trading_bot",
                status=HealthStatus.HEALTHY,
                message="Trading bot is running normally",
                timestamp=datetime.now(),
                details={
                    'uptime': (datetime.now() - self.bot_monitor.start_time).total_seconds(),
                    'trade_count': self.bot_monitor.trade_count,
                    'error_count': self.bot_monitor.error_count
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="trading_bot",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check bot health: {str(e)}",
                timestamp=datetime.now()
            )
    
    async def run_all_checks(self) -> List[HealthCheck]:
        """Run all health checks."""
        checks = []
        
        # Database health check
        checks.append(await self.check_database_health())
        
        # System resources check
        checks.append(await self.check_system_resources())
        
        # Bot health check
        checks.append(await self.check_bot_health())
        
        return checks
    
    def get_overall_status(self, checks: List[HealthCheck]) -> HealthStatus:
        """Get overall health status from individual checks."""
        if any(check.status == HealthStatus.CRITICAL for check in checks):
            return HealthStatus.CRITICAL
        elif any(check.status == HealthStatus.WARNING for check in checks):
            return HealthStatus.WARNING
        elif all(check.status == HealthStatus.HEALTHY for check in checks):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN


class MetricsCollector:
    """Prometheus-style metrics collection."""
    
    def __init__(self):
        self.metrics = {}
        self.logger = logging.getLogger(__name__)
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment a counter metric."""
        if name not in self.metrics:
            self.metrics[name] = {'type': 'counter', 'value': 0.0, 'labels': labels or {}}
        
        self.metrics[name]['value'] += value
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric."""
        self.metrics[name] = {
            'type': 'gauge',
            'value': value,
            'labels': labels or {},
            'timestamp': time.time()
        }
    
    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Observe a histogram metric."""
        if name not in self.metrics:
            self.metrics[name] = {
                'type': 'histogram',
                'values': [],
                'labels': labels or {}
            }
        
        self.metrics[name]['values'].append({
            'value': value,
            'timestamp': time.time()
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics in Prometheus format."""
        return self.metrics
    
    def format_prometheus(self) -> str:
        """Format metrics in Prometheus text format."""
        lines = []
        
        for name, metric in self.metrics.items():
            if metric['type'] == 'counter':
                labels_str = ','.join([f'{k}="{v}"' for k, v in metric['labels'].items()])
                if labels_str:
                    lines.append(f'{name}{{{labels_str}}} {metric["value"]}')
                else:
                    lines.append(f'{name} {metric["value"]}')
            
            elif metric['type'] == 'gauge':
                labels_str = ','.join([f'{k}="{v}"' for k, v in metric['labels'].items()])
                if labels_str:
                    lines.append(f'{name}{{{labels_str}}} {metric["value"]}')
                else:
                    lines.append(f'{name} {metric["value"]}')
            
            elif metric['type'] == 'histogram':
                if metric['values']:
                    avg_value = sum(v['value'] for v in metric['values']) / len(metric['values'])
                    labels_str = ','.join([f'{k}="{v}"' for k, v in metric['labels'].items()])
                    if labels_str:
                        lines.append(f'{name}_avg{{{labels_str}}} {avg_value}')
                        lines.append(f'{name}_count{{{labels_str}}} {len(metric["values"])}')
                    else:
                        lines.append(f'{name}_avg {avg_value}')
                        lines.append(f'{name}_count {len(metric["values"])}')
        
        return '\n'.join(lines)


class ProductionManager:
    """Production environment manager."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.bot_monitor = TradingBotMonitor(db_manager)
        self.health_checker = HealthChecker(db_manager, self.bot_monitor)
        self.metrics_collector = MetricsCollector()
        self.logger = logging.getLogger(__name__)
        self.is_production = self._detect_production_environment()
    
    def _detect_production_environment(self) -> bool:
        """Detect if running in production environment."""
        return (
            os.getenv('PRODUCTION', 'false').lower() == 'true' or
            os.getenv('ENVIRONMENT', 'development').lower() == 'production'
        )
    
    async def start_monitoring(self):
        """Start production monitoring."""
        if not self.is_production:
            self.logger.info("Not in production environment, skipping monitoring")
            return
        
        self.logger.info("Starting production monitoring")
        
        # Start background monitoring tasks
        asyncio.create_task(self._monitoring_loop())
        asyncio.create_task(self._metrics_collection_loop())
        asyncio.create_task(self._health_check_loop())
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while True:
            try:
                # Update system metrics
                system_monitor = SystemMonitor()
                cpu_usage = system_monitor.get_cpu_usage()
                memory_usage = system_monitor.get_memory_usage()
                disk_usage = system_monitor.get_disk_usage()
                
                # Update metrics
                self.metrics_collector.set_gauge('system_cpu_usage', cpu_usage)
                self.metrics_collector.set_gauge('system_memory_usage', memory_usage['percentage'])
                self.metrics_collector.set_gauge('system_disk_usage', disk_usage['percentage'])
                
                # Update bot heartbeat
                self.bot_monitor.update_heartbeat()
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_collection_loop(self):
        """Background metrics collection loop."""
        while True:
            try:
                # Collect trading metrics
                trading_metrics = await self.bot_monitor.get_trading_metrics()
                position_metrics = await self.bot_monitor.get_position_metrics()
                
                # Update metrics
                self.metrics_collector.set_gauge('trading_total_trades', trading_metrics.get('total_trades', 0))
                self.metrics_collector.set_gauge('trading_win_rate', trading_metrics.get('win_rate', 0.0))
                self.metrics_collector.set_gauge('trading_total_pnl', trading_metrics.get('total_pnl', 0.0))
                self.metrics_collector.set_gauge('trading_total_positions', position_metrics.get('total_positions', 0))
                self.metrics_collector.set_gauge('trading_unrealized_pnl', position_metrics.get('unrealized_pnl', 0.0))
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(60)
    
    async def _health_check_loop(self):
        """Background health check loop."""
        while True:
            try:
                # Run health checks
                checks = await self.health_checker.run_all_checks()
                overall_status = self.health_checker.get_overall_status(checks)
                
                # Update metrics
                self.metrics_collector.set_gauge('health_status', 1 if overall_status == HealthStatus.HEALTHY else 0)
                
                # Log critical issues
                for check in checks:
                    if check.status == HealthStatus.CRITICAL:
                        self.logger.critical(f"Critical health check failure: {check.name} - {check.message}")
                    elif check.status == HealthStatus.WARNING:
                        self.logger.warning(f"Health check warning: {check.name} - {check.message}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(300)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        checks = await self.health_checker.run_all_checks()
        overall_status = self.health_checker.get_overall_status(checks)
        
        return {
            'overall_status': overall_status.value,
            'timestamp': datetime.now().isoformat(),
            'checks': [
                {
                    'name': check.name,
                    'status': check.status.value,
                    'message': check.message,
                    'timestamp': check.timestamp.isoformat(),
                    'details': check.details
                }
                for check in checks
            ],
            'bot_status': await self.bot_monitor.get_bot_status(),
            'system_metrics': self.metrics_collector.get_metrics()
        }
    
    async def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return self.metrics_collector.format_prometheus()
    
    def increment_trade_count(self):
        """Increment trade count."""
        self.bot_monitor.increment_trade_count()
        self.metrics_collector.increment_counter('trades_total')
    
    def increment_error_count(self):
        """Increment error count."""
        self.bot_monitor.increment_error_count()
        self.metrics_collector.increment_counter('errors_total')


# Health check endpoint for web frameworks
async def health_check_endpoint(production_manager: ProductionManager):
    """Health check endpoint for web frameworks."""
    try:
        health_status = await production_manager.get_health_status()
        
        # Determine HTTP status code
        if health_status['overall_status'] == 'healthy':
            status_code = 200
        elif health_status['overall_status'] == 'warning':
            status_code = 200  # Still OK for load balancers
        else:
            status_code = 503  # Service unavailable
        
        return {
            'status_code': status_code,
            'data': health_status
        }
        
    except Exception as e:
        return {
            'status_code': 503,
            'data': {
                'overall_status': 'critical',
                'message': f'Health check failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
        }


# Metrics endpoint for Prometheus
async def metrics_endpoint(production_manager: ProductionManager):
    """Metrics endpoint for Prometheus scraping."""
    try:
        metrics = await production_manager.get_metrics()
        return {
            'status_code': 200,
            'content_type': 'text/plain; version=0.0.4; charset=utf-8',
            'data': metrics
        }
    except Exception as e:
        return {
            'status_code': 500,
            'content_type': 'text/plain',
            'data': f'# Error generating metrics: {str(e)}\n'
        }
