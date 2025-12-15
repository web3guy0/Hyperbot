/**
 * PM2 Ecosystem Configuration
 * Manages HyperBot with auto-restart, monitoring, and logging
 * 
 * Usage:
 *   pm2 start ecosystem.config.js
 *   pm2 logs hyperbot
 *   pm2 monit
 *   pm2 restart hyperbot
 *   pm2 stop hyperbot
 */

module.exports = {
  apps: [{
    name: 'hyperbot',
    script: 'app/bot.py',
    interpreter: 'python3',
    
    // Auto-restart configuration
    autorestart: true,
    max_restarts: 10,
    min_uptime: '10s',
    restart_delay: 5000,
    
    // Memory management
    max_memory_restart: '500M',
    
    // Error handling
    error_file: 'logs/pm2-error.log',
    out_file: 'logs/pm2-out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    merge_logs: true,
    
    // Environment variables (load from .env)
    env: {
      NODE_ENV: 'production',
      PYTHONUNBUFFERED: '1'
    },
    
    // Restart conditions
    exp_backoff_restart_delay: 100,
    
    // Monitoring
    instances: 1,
    exec_mode: 'fork',
    
    // Cron restart (optional: restart daily at 4am UTC)
    cron_restart: '0 4 * * *',
    
    // Watch (disabled for production)
    watch: false,
    
    // Kill timeout
    kill_timeout: 10000,
    
    // Listen timeout
    listen_timeout: 10000
  }]
}
