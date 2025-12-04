"""
Risk Guardian - Kill Switch Risk Management System
Monitors and protects trading capital with multiple safety layers.
"""

import MetaTrader5 as mt5
import json
import os
from datetime import datetime, date
from typing import Tuple


class RiskGuardian:
    """
    Kill Switch Risk Management System
    
    3 Protection Levels:
    1. Daily Loss Limit - Stop trading if loss > X% today
    2. Drawdown Monitor - Reduce lot size if balance drops X% from peak
    3. Volatility Guard - Pause if spread is too high
    """
    
    def __init__(self, config: dict, state_file: str = "risk_state.json"):
        # Load config
        kill_switch_config = config.get("kill_switch", {})
        
        self.daily_loss_limit = kill_switch_config.get("daily_loss_limit", 0.02)  # 2%
        self.max_drawdown = kill_switch_config.get("max_drawdown", 0.05)  # 5%
        self.volatility_spread = kill_switch_config.get("volatility_spread", 50)  # points
        self.lot_reduction = kill_switch_config.get("lot_reduction", 0.5)  # 50%
        
        # State file path
        self.state_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            kill_switch_config.get("state_file", "risk_state.json")
        )
        
        # Load or initialize state
        self.state = self._load_state()
        
        # Check for daily reset
        self._check_daily_reset()
        
        print(f"🛡️ Risk Guardian initialized")
        print(f"   Daily Loss Limit: {self.daily_loss_limit*100:.1f}%")
        print(f"   Max Drawdown: {self.max_drawdown*100:.1f}%")
        print(f"   Volatility Spread: {self.volatility_spread} points")
    
    def _load_state(self) -> dict:
        """Load state from file or create default"""
        default_state = {
            "peak_balance": 0.0,
            "daily_start_balance": 0.0,
            "last_reset_date": str(date.today()),
            "kill_switch_active": False,
            "lot_reduction_active": False
        }
        
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r") as f:
                    loaded = json.load(f)
                    # Merge with defaults to handle missing keys
                    return {**default_state, **loaded}
        except Exception as e:
            print(f"⚠️ Error loading risk state: {e}")
        
        return default_state
    
    def _save_state(self):
        """Persist state to file"""
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving risk state: {e}")
    
    def _check_daily_reset(self):
        """Reset daily counters at midnight"""
        today = str(date.today())
        
        if self.state["last_reset_date"] != today:
            print(f"📅 New trading day - Resetting daily limits")
            
            # Get current balance for new day
            account = mt5.account_info()
            if account:
                current_balance = account.balance
                self.state["daily_start_balance"] = current_balance
                
                # Update peak if current is higher
                if current_balance > self.state["peak_balance"]:
                    self.state["peak_balance"] = current_balance
            
            # Reset kill switch
            self.state["kill_switch_active"] = False
            self.state["last_reset_date"] = today
            
            self._save_state()
    
    def update_balances(self) -> Tuple[float, float, float]:
        """
        Update and return current balance metrics.
        Returns: (current_balance, daily_pnl_percent, drawdown_percent)
        """
        account = mt5.account_info()
        if not account:
            return 0.0, 0.0, 0.0
        
        current_balance = account.balance
        
        # Initialize if first run
        if self.state["peak_balance"] == 0:
            self.state["peak_balance"] = current_balance
            self.state["daily_start_balance"] = current_balance
            self._save_state()
        
        # Update peak balance (high water mark)
        if current_balance > self.state["peak_balance"]:
            self.state["peak_balance"] = current_balance
            self.state["lot_reduction_active"] = False  # Reset reduction
            self._save_state()
        
        # Calculate metrics
        daily_start = self.state["daily_start_balance"]
        peak = self.state["peak_balance"]
        
        daily_pnl_percent = (current_balance - daily_start) / daily_start if daily_start > 0 else 0
        drawdown_percent = (peak - current_balance) / peak if peak > 0 else 0
        
        return current_balance, daily_pnl_percent, drawdown_percent
    
    def can_trade(self) -> Tuple[bool, str]:
        """
        Main check - Can we trade right now?
        Returns: (allowed, reason)
        """
        # Check if kill switch is already active
        if self.state["kill_switch_active"]:
            return False, "Kill switch active until tomorrow"
        
        # Check daily reset
        self._check_daily_reset()
        
        # Get current metrics
        balance, daily_pnl, drawdown = self.update_balances()
        
        # Check Daily Loss Limit
        if daily_pnl < -self.daily_loss_limit:
            self.state["kill_switch_active"] = True
            self._save_state()
            return False, f"Daily loss limit hit ({daily_pnl*100:.2f}%)"
        
        return True, "OK"
    
    def get_lot_multiplier(self) -> float:
        """
        Returns lot size multiplier based on drawdown.
        1.0 = normal, 0.5 = reduced (in drawdown)
        """
        balance, daily_pnl, drawdown = self.update_balances()
        
        # Check drawdown threshold
        if drawdown >= self.max_drawdown:
            if not self.state["lot_reduction_active"]:
                print(f"⚠️ DRAWDOWN ALERT: {drawdown*100:.2f}% from peak - Reducing lot size")
                self.state["lot_reduction_active"] = True
                self._save_state()
            return self.lot_reduction
        
        return 1.0
    
    def check_volatility(self, spread_points: float) -> Tuple[bool, str]:
        """
        Check if spread is acceptable for trading.
        Returns: (acceptable, reason)
        """
        if spread_points > self.volatility_spread:
            return False, f"Spread too high: {spread_points:.0f} > {self.volatility_spread}"
        
        return True, "OK"
    
    def get_status(self) -> dict:
        """Get current guardian status for dashboard"""
        balance, daily_pnl, drawdown = self.update_balances()
        
        return {
            "balance": balance,
            "peak_balance": self.state["peak_balance"],
            "daily_pnl_percent": daily_pnl * 100,
            "drawdown_percent": drawdown * 100,
            "kill_switch_active": self.state["kill_switch_active"],
            "lot_reduction_active": self.state["lot_reduction_active"],
            "lot_multiplier": self.get_lot_multiplier()
        }
    
    def force_kill_switch(self, reason: str = "Manual activation"):
        """Manually activate kill switch"""
        print(f"🛑 KILL SWITCH ACTIVATED: {reason}")
        self.state["kill_switch_active"] = True
        self._save_state()
    
    def reset_kill_switch(self):
        """Manually reset kill switch (use with caution)"""
        print("♻️ Kill switch manually reset")
        self.state["kill_switch_active"] = False
        self._save_state()


# Standalone test
if __name__ == "__main__":
    import yaml
    
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize MT5
    if not mt5.initialize():
        print("MT5 init failed")
        exit()
    
    # Test guardian
    guardian = RiskGuardian(config)
    
    can_trade, reason = guardian.can_trade()
    print(f"Can trade: {can_trade} - {reason}")
    
    lot_mult = guardian.get_lot_multiplier()
    print(f"Lot multiplier: {lot_mult}")
    
    status = guardian.get_status()
    print(f"Status: {json.dumps(status, indent=2)}")
    
    mt5.shutdown()
