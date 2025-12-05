"""
Notification system for trading alerts.
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
import requests


logger = logging.getLogger(__name__)


class NotificationManager:
    """
    Manages notifications via multiple channels (Telegram, Email, etc.).
    """
    
    def __init__(
        self,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        email_enabled: bool = False,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        email_to: Optional[str] = None
    ):
        """
        Initialize notification manager.
        
        Args:
            telegram_token: Telegram bot token
            telegram_chat_id: Telegram chat ID
            email_enabled: Whether email notifications are enabled
            smtp_host: SMTP server host
            smtp_port: SMTP server port
            smtp_user: SMTP username
            smtp_password: SMTP password
            email_to: Recipient email address
        """
        # Telegram configuration
        self.telegram_enabled = bool(telegram_token and telegram_chat_id)
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        
        # Email configuration
        self.email_enabled = email_enabled
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.email_to = email_to
        
        logger.info(
            f"NotificationManager initialized: "
            f"Telegram={'enabled' if self.telegram_enabled else 'disabled'}, "
            f"Email={'enabled' if self.email_enabled else 'disabled'}"
        )
    
    def send_telegram(self, message: str) -> bool:
        """
        Send message via Telegram.
        
        Args:
            message: Message text
            
        Returns:
            True if successful
        """
        if not self.telegram_enabled:
            logger.debug("Telegram not configured, skipping notification")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            logger.debug("Telegram notification sent")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False
    
    def send_email(self, subject: str, body: str) -> bool:
        """
        Send email notification.
        
        Args:
            subject: Email subject
            body: Email body
            
        Returns:
            True if successful
        """
        if not self.email_enabled:
            logger.debug("Email not configured, skipping notification")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_user
            msg['To'] = self.email_to
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            logger.debug("Email notification sent")
            return True
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    def notify_trade(self, action: str, symbol: str, side: str, quantity: float, price: float):
        """
        Send trade notification.
        
        Args:
            action: "OPEN" or "CLOSE"
            symbol: Trading pair
            side: "Buy" or "Sell"
            quantity: Trade quantity
            price: Execution price
        """
        message = (
            f"🤖 <b>Trade {action}</b>\n"
            f"Symbol: {symbol}\n"
            f"Side: {side}\n"
            f"Quantity: {quantity}\n"
            f"Price: ${price:.2f}"
        )
        
        self.send_telegram(message)
    
    def notify_error(self, error_msg: str):
        """
        Send error notification.
        
        Args:
            error_msg: Error message
        """
        message = f"⚠️ <b>ERROR</b>\n{error_msg}"
        self.send_telegram(message)
    
    def notify_bot_status(self, status: str, details: str = ""):
        """
        Send bot status notification.
        
        Args:
            status: "STARTED", "STOPPED", "ERROR"
            details: Additional details
        """
        emoji = "✅" if status == "STARTED" else "❌"
        message = f"{emoji} <b>Bot {status}</b>\n{details}"
        self.send_telegram(message)
    
    def notify_daily_summary(self, stats: dict):
        """
        Send daily performance summary.
        
        Args:
            stats: Performance statistics dictionary
        """
        message = (
            f"📊 <b>Daily Summary</b>\n"
            f"Trades: {stats.get('total_trades', 0)}\n"
            f"Win Rate: {stats.get('win_rate', 0):.1f}%\n"
            f"PnL: ${stats.get('net_pnl', 0):+.2f}\n"
            f"Balance: ${stats.get('current_balance', 0):.2f}"
        )
        
        self.send_telegram(message)


# TODO: Add Telegram bot token and chat ID in .env file
# Create bot: https://t.me/BotFather
# Get chat ID: Send message to bot, then visit:
# https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
