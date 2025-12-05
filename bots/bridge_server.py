"""
Python Bridge Server for MQL5 Trading
Acts as a TCP server that the MQL5 EA connects to.
Receives tick data, sends trading commands.

Usage:
    python bots/bridge_server.py
"""

import socket
import threading
import json
import queue
import time
from datetime import datetime
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
HOST = '127.0.0.1'
PORT = 5555
BUFFER_SIZE = 4096


class TradingBridge:
    """
    TCP Server that communicates with MQL5 EA.
    Provides low-latency order execution and tick data streaming.
    """
    
    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.client_address = None
        self.connected = False
        self.running = False
        
        # Queues for async communication
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.tick_queue = queue.Queue(maxsize=100)  # Limited size for ticks
        
        # Latest tick data
        self.last_tick = None
        
        # Callbacks
        self.on_tick_callback = None
        self.on_execution_callback = None
        self.on_connect_callback = None
        self.on_disconnect_callback = None
        
        # Threads
        self.accept_thread = None
        self.receive_thread = None
        self.send_thread = None
    
    def start(self):
        """Start the bridge server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        
        self.running = True
        print(f"🌉 Trading Bridge Server started on {self.host}:{self.port}")
        print("⏳ Waiting for MQL5 EA to connect...")
        
        # Start accept thread
        self.accept_thread = threading.Thread(target=self._accept_connection, daemon=True)
        self.accept_thread.start()
        
        # Start send thread
        self.send_thread = threading.Thread(target=self._send_commands, daemon=True)
        self.send_thread.start()
    
    def stop(self):
        """Stop the bridge server"""
        self.running = False
        self.connected = False
        
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        print("🛑 Trading Bridge Server stopped")
    
    def _accept_connection(self):
        """Accept incoming connection from MQL5 EA"""
        while self.running:
            try:
                self.server_socket.settimeout(1.0)
                client, address = self.server_socket.accept()
                
                self.client_socket = client
                self.client_address = address
                self.connected = True
                
                print(f"✅ MQL5 EA connected from {address}")
                
                if self.on_connect_callback:
                    self.on_connect_callback(address)
                
                # Start receive thread for this client
                self.receive_thread = threading.Thread(target=self._receive_messages, daemon=True)
                self.receive_thread.start()
                
                # Wait for disconnection
                self.receive_thread.join()
                
                self.connected = False
                print(f"❌ MQL5 EA disconnected")
                
                if self.on_disconnect_callback:
                    self.on_disconnect_callback()
                    
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Accept error: {e}")
    
    def _receive_messages(self):
        """Receive messages from MQL5 EA"""
        buffer = ""
        
        while self.running and self.connected:
            try:
                self.client_socket.settimeout(0.1)
                data = self.client_socket.recv(BUFFER_SIZE)
                
                if not data:
                    break
                
                buffer += data.decode('utf-8')
                
                # Process complete messages (newline-delimited)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        self._process_message(line.strip())
                        
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Receive error: {e}")
                break
    
    def _process_message(self, message):
        """Process a message from MQL5 EA"""
        try:
            data = json.loads(message)
            msg_type = data.get('type', '')
            
            if msg_type == 'tick':
                self.last_tick = data
                if not self.tick_queue.full():
                    self.tick_queue.put(data)
                if self.on_tick_callback:
                    self.on_tick_callback(data)
            
            elif msg_type == 'candle':
                # Incoming historical candle
                if hasattr(self, '_history_buffer'):
                    self._history_buffer.append(data)
                    
            elif msg_type == 'history_start':
                self._history_buffer = []
                print(f"📥 Receiving history: {data.get('count')} candles...")
                
            elif msg_type == 'history_end':
                print(f"✅ History received: {len(self._history_buffer)} candles")
                self.response_queue.put({'type': 'history_data', 'data': self._history_buffer})
                delattr(self, '_history_buffer')
                    
            elif msg_type == 'execution':
                self.response_queue.put(data)
                if self.on_execution_callback:
                    self.on_execution_callback(data)
                print(f"📊 Execution: {data.get('action')} - {data.get('status')}")
                
            elif msg_type == 'handshake':
                print(f"🤝 Handshake: EA={data.get('ea')}, Symbol={data.get('symbol')}")
                
            elif msg_type == 'positions':
                self.response_queue.put(data)
                
            elif msg_type == 'account':
                self.response_queue.put(data)
                
            elif msg_type == 'pong':
                latency = (datetime.now() - self._ping_time).total_seconds() * 1000
                print(f"📶 Ping: {latency:.1f}ms")
                
            elif msg_type == 'close':
                self.response_queue.put(data)
                
            elif msg_type == 'close_all':
                self.response_queue.put(data)
                
            else:
                print(f"Unknown message type: {msg_type}")
                
        except json.JSONDecodeError:
            print(f"Invalid JSON: {message}")
    
    def _send_commands(self):
        """Send commands to MQL5 EA"""
        while self.running:
            try:
                command = self.command_queue.get(timeout=0.1)
                if self.connected and self.client_socket:
                    message = json.dumps(command) + '\n'
                    self.client_socket.send(message.encode('utf-8'))
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Send error: {e}")
    
    # ========================================
    # Public API - Trading Commands
    # ========================================
    
    def buy(self, volume=0.01, sl_points=0, tp_points=0, comment=""):
        """Send BUY order to MQL5 EA"""
        command = {
            "action": "BUY",
            "volume": volume,
            "sl_points": sl_points,
            "tp_points": tp_points,
            "comment": comment
        }
        self.command_queue.put(command)
        print(f"📤 Sent BUY command: {volume} lots")
    
    def sell(self, volume=0.01, sl_points=0, tp_points=0, comment=""):
        """Send SELL order to MQL5 EA"""
        command = {
            "action": "SELL",
            "volume": volume,
            "sl_points": sl_points,
            "tp_points": tp_points,
            "comment": comment
        }
        self.command_queue.put(command)
        print(f"📤 Sent SELL command: {volume} lots")
    
    def close_position(self, ticket):
        """Close specific position"""
        command = {
            "action": "CLOSE",
            "ticket": ticket
        }
        self.command_queue.put(command)
    
    def modify_position(self, ticket, sl, tp):
        """Modify position SL/TP"""
        command = {
            "action": "MODIFY",
            "ticket": ticket,
            "sl": float(sl),
            "tp": float(tp)
        }
        self.command_queue.put(command)
    
    def close_all(self):
        """Close all positions"""
        command = {"action": "CLOSE_ALL"}
        self.command_queue.put(command)
    
    def get_positions(self):
        """Request current positions from EA"""
        command = {"action": "GET_POSITIONS"}
        self.command_queue.put(command)
    
    def get_account(self):
        """Request account info from EA"""
        command = {"action": "GET_ACCOUNT"}
        self.command_queue.put(command)
        
    def get_history(self, count=500):
        """Request historical M5 candles"""
        command = {"action": "GET_HISTORY", "count": count}
        self.command_queue.put(command)
    
    def ping(self):
        """Send ping to measure latency"""
        self._ping_time = datetime.now()
        command = {"action": "PING"}
        self.command_queue.put(command)
    
    def get_tick(self):
        """Get the latest tick data"""
        return self.last_tick
    
    def wait_response(self, timeout=5.0):
        """Wait for a response from EA"""
        try:
            return self.response_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# ========================================
# Main - Test the bridge
# ========================================

if __name__ == "__main__":
    bridge = TradingBridge()
    
    # Set up callbacks
    def on_tick(tick):
        # Print tick every 5 seconds to avoid spam
        pass
    
    def on_execution(result):
        print(f"🎯 Trade Result: {result}")
    
    def on_connect(addr):
        print(f"🔗 Connected: {addr}")
    
    bridge.on_tick_callback = on_tick
    bridge.on_execution_callback = on_execution
    bridge.on_connect_callback = on_connect
    
    try:
        bridge.start()
        
        print("\n" + "="*50)
        print("Trading Bridge Server Running")
        print("="*50)
        print("Commands:")
        print("  b - Send BUY order")
        print("  s - Send SELL order")
        print("  c - Close all positions")
        print("  p - Ping (measure latency)")
        print("  t - Show last tick")
        print("  a - Get account info")
        print("  q - Quit")
        print("="*50 + "\n")
        
        while True:
            try:
                cmd = input("> ").strip().lower()
                
                if cmd == 'q':
                    break
                elif cmd == 'b':
                    volume = input("Volume (default 0.01): ").strip()
                    volume = float(volume) if volume else 0.01
                    sl = input("SL points (default 800): ").strip()
                    sl = float(sl) if sl else 800
                    bridge.buy(volume=volume, sl_points=sl, comment="ManualTest")
                elif cmd == 's':
                    volume = input("Volume (default 0.01): ").strip()
                    volume = float(volume) if volume else 0.01
                    sl = input("SL points (default 800): ").strip()
                    sl = float(sl) if sl else 800
                    bridge.sell(volume=volume, sl_points=sl, comment="ManualTest")
                elif cmd == 'c':
                    bridge.close_all()
                elif cmd == 'p':
                    bridge.ping()
                elif cmd == 't':
                    tick = bridge.get_tick()
                    if tick:
                        print(f"Bid: {tick.get('bid')}, Ask: {tick.get('ask')}, Spread: {tick.get('spread')}")
                    else:
                        print("No tick data yet")
                elif cmd == 'a':
                    bridge.get_account()
                    time.sleep(0.5)
                    response = bridge.wait_response(timeout=2)
                    if response:
                        print(f"Balance: {response.get('balance')}, Equity: {response.get('equity')}")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                
    finally:
        bridge.stop()
