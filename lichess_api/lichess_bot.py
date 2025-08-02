#!/usr/bin/env python3
"""
Lichess Bot for ChessGPT using NanoGPT models
"""

import requests
import json
import threading
import time
import chess
import chess.pgn
import io
from typing import Optional, Dict, Any
import sys
import os

# We'll implement our own simple model interface

class LichessAPI:
    def __init__(self, token: str):
        self.token = token
        self.headers = {"Authorization": f"Bearer {token}"}
        self.base_url = "https://lichess.org/api"
    
    def upgrade_to_bot(self):
        """Upgrade account to bot (irreversible!)"""
        response = requests.post(f"{self.base_url}/bot/account/upgrade", headers=self.headers)
        return response.status_code == 200, response.text
    
    def accept_challenge(self, challenge_id: str):
        """Accept a challenge"""
        response = requests.post(f"{self.base_url}/challenge/{challenge_id}/accept", headers=self.headers)
        return response.status_code == 200
    
    def decline_challenge(self, challenge_id: str, reason: str = "generic"):
        """Decline a challenge"""
        response = requests.post(
            f"{self.base_url}/challenge/{challenge_id}/decline",
            headers=self.headers,
            json={"reason": reason}
        )
        return response.status_code == 200
    
    def make_move(self, game_id: str, move: str):
        """Make a move in UCI format"""
        response = requests.post(
            f"{self.base_url}/bot/game/{game_id}/move/{move}",
            headers=self.headers
        )
        return response.status_code == 200, response.text
    
    def stream_events(self):
        """Stream bot events (challenges, game starts)"""
        response = requests.get(
            f"{self.base_url}/stream/event",
            headers=self.headers,
            stream=True
        )
        for line in response.iter_lines():
            if line:
                try:
                    yield json.loads(line.decode('utf-8'))
                except json.JSONDecodeError:
                    continue
    
    def stream_game(self, game_id: str):
        """Stream game state updates"""
        response = requests.get(
            f"{self.base_url}/bot/game/stream/{game_id}",
            headers=self.headers,
            stream=True
        )
        for line in response.iter_lines():
            if line:
                try:
                    yield json.loads(line.decode('utf-8'))
                except json.JSONDecodeError:
                    continue

class ChessGPTBot:
    def __init__(self, token: str, model_name: str = "large-16-ckpt.pt"):
        self.api = LichessAPI(token)
        self.model_name = model_name
        self.nanogpt_player = None
        self.active_games = {}
        self.bot_username = "Chess-GPT2"  # Your bot's username
        
        # Initialize model
        try:
            # TODO: Implement clean model loading
            print(f"✅ Model ready: {model_name}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            sys.exit(1)
    
    def start(self):
        """Start the bot"""
        print(f"🤖 Starting ChessGPT Bot ({self.model_name})")
        print("📡 Listening for challenges and games...")
        
        try:
            for event in self.api.stream_events():
                self.handle_event(event)
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
        except Exception as e:
            print(f"❌ Bot error: {e}")
    
    def handle_event(self, event: Dict[str, Any]):
        """Handle incoming events"""
        event_type = event.get('type')
        
        if event_type == 'challenge':
            self.handle_challenge(event['challenge'])
        elif event_type == 'gameStart':
            self.handle_game_start(event['game'])
        elif event_type == 'gameFinish':
            self.handle_game_finish(event['game'])
    
    def handle_challenge(self, challenge: Dict[str, Any]):
        """Handle incoming challenges"""
        challenge_id = challenge['id']
        challenger = challenge['challenger']['name']
        variant = challenge['variant']['name']
        time_control = challenge['timeControl']
        color = challenge.get('color', 'random')
        
        print(f"🎯 Challenge from {challenger}")
        print(f"   Variant: {variant}")
        print(f"   Time: {time_control}")
        print(f"   Color: {color}")
        
        # Accept criteria
        should_accept = True
        decline_reason = "generic"
        
        # Only accept standard chess
        if variant != "Standard":
            should_accept = False
            decline_reason = "variant"
            print(f"   ❌ Declining: Only play Standard chess")
        
        # Only play as White (decline if forced to play Black)
        elif color == "black":
            should_accept = False
            decline_reason = "generic"
            print(f"   ❌ Declining: Only play as White")
        
        # Accept reasonable time controls (avoid bullet games)
        elif time_control.get('type') == 'clock':
            initial_time = time_control.get('limit', 0)  # seconds
            if initial_time < 180:  # Less than 3 minutes
                should_accept = False
                decline_reason = "timeControl"
                print(f"   ❌ Declining: Time control too fast")
        
        if should_accept:
            if self.api.accept_challenge(challenge_id):
                print(f"   ✅ Accepted challenge from {challenger}")
            else:
                print(f"   ❌ Failed to accept challenge")
        else:
            if self.api.decline_challenge(challenge_id, decline_reason):
                print(f"   ❌ Declined challenge from {challenger}")
    
    def handle_game_start(self, game: Dict[str, Any]):
        """Handle game start"""
        game_id = game['id']
        print(f"🎮 Game started: {game_id}")
        
        # Start game in separate thread
        game_thread = threading.Thread(target=self.play_game, args=(game_id,))
        game_thread.daemon = True
        game_thread.start()
    
    def handle_game_finish(self, game: Dict[str, Any]):
        """Handle game finish"""
        game_id = game['id']
        if game_id in self.active_games:
            del self.active_games[game_id]
        print(f"🏁 Game finished: {game_id}")
    
    def play_game(self, game_id: str):
        """Play a single game"""
        print(f"🎯 Playing game: {game_id}")
        
        game_state = {
            'moves': '',
            'pgn_moves': [],
            'board': chess.Board(),
            'my_color': None,
            'my_turn': False
        }
        
        self.active_games[game_id] = game_state
        
        try:
            for event in self.api.stream_game(game_id):
                if not self.process_game_event(game_id, event):
                    break
        except Exception as e:
            print(f"❌ Error in game {game_id}: {e}")
        finally:
            if game_id in self.active_games:
                del self.active_games[game_id]
    
    def process_game_event(self, game_id: str, event: Dict[str, Any]) -> bool:
        """Process a single game event. Returns False if game should end."""
        if game_id not in self.active_games:
            return False
        
        game_state = self.active_games[game_id]
        event_type = event.get('type')
        
        if event_type == 'gameFull':
            # Initial game state
            white_player = event['white']['name']
            black_player = event['black']['name']
            
            # Determine our color
            if white_player == self.bot_username:
                game_state['my_color'] = chess.WHITE
                print(f"   Playing as White vs {black_player}")
            elif black_player == self.bot_username:
                game_state['my_color'] = chess.BLACK
                print(f"   Playing as Black vs {white_player}")
            else:
                print(f"   ❌ Bot not found in game players")
                return False
            
            # Process initial moves
            initial_moves = event['state']['moves']
            self.update_game_state(game_state, initial_moves)
            
        elif event_type == 'gameState':
            # Game state update
            moves = event['moves']
            self.update_game_state(game_state, moves)
            
        elif event_type == 'chatLine':
            # Chat message (ignore for now)
            pass
        
        # Make move if it's our turn
        if game_state['my_turn'] and event_type in ['gameFull', 'gameState']:
            self.make_bot_move(game_id, game_state)
        
        return True
    
    def update_game_state(self, game_state: Dict[str, Any], moves_string: str):
        """Update game state with new moves"""
        if moves_string == game_state['moves']:
            return  # No new moves
        
        game_state['moves'] = moves_string
        
        # Reset board and replay all moves
        game_state['board'] = chess.Board()
        game_state['pgn_moves'] = []
        
        if moves_string.strip():
            uci_moves = moves_string.strip().split()
            for uci_move in uci_moves:
                try:
                    move = chess.Move.from_uci(uci_move)
                    san_move = game_state['board'].san(move)
                    game_state['board'].push(move)
                    game_state['pgn_moves'].append(san_move)
                except Exception as e:
                    print(f"❌ Error parsing move {uci_move}: {e}")
                    break
        
        # Determine if it's our turn
        current_turn = game_state['board'].turn
        game_state['my_turn'] = (current_turn == game_state['my_color'])
        
        print(f"   Moves: {len(game_state['pgn_moves'])}, My turn: {game_state['my_turn']}")
    
    def make_bot_move(self, game_id: str, game_state: Dict[str, Any]):
        """Make a move using NanoGPT"""
        try:
            print(f"   🤔 Thinking... (Moves: {len(game_state['pgn_moves'])})")
            
            # Get move from model
            board_copy = game_state['board'].copy()
            san_move = self.get_model_move(board_copy, game_state['pgn_moves'])
            
            if not san_move:
                print(f"   ❌ Model returned no move")
                return
            
            # Clean up the move (remove leading space if present)
            san_move = san_move.strip()
            
            # Convert SAN to UCI
            try:
                move = board_copy.parse_san(san_move)
                uci_move = move.uci()
                
                print(f"   🎯 Playing: {san_move} ({uci_move})")
                
                # Make the move on Lichess
                success, response = self.api.make_move(game_id, uci_move)
                if success:
                    print(f"   ✅ Move sent successfully")
                    game_state['my_turn'] = False
                else:
                    print(f"   ❌ Failed to send move: {response}")
                    
            except Exception as e:
                print(f"   ❌ Invalid move from model: {san_move} - {e}")
                
        except Exception as e:
            print(f"   ❌ Error making move: {e}")
    
    def get_model_move(self, board: chess.Board, pgn_moves: list) -> str:
        """Get move from API server"""
        try:
            response = requests.post('http://localhost:8000/api/move', json={
                'fen': board.fen(),
                'pgn': pgn_moves,
                'engine': 'nanogpt',
                'model': self.model_name.replace('-ckpt.pt', ''),  # e.g., "large-16"
                'temperature': 0.1
            }, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('move')
            else:
                print(f"   ❌ API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"   ❌ API timeout")
            return None
        except Exception as e:
            print(f"   ❌ API request failed: {e}")
            return None
    
    def format_pgn_for_nanogpt(self, pgn_moves: list) -> str:
        """Format PGN moves for NanoGPT (matching main.py format)"""
        if not pgn_moves:
            # Read the initial prompt from file (like main.py does)
            try:
                with open("gpt_inputs/prompt.txt", "r") as f:
                    return f.read()
            except:
                return ""  # Fallback to empty string
        
        # Build PGN string like main.py does
        game_state = ""
        
        # Add initial prompt
        try:
            with open("gpt_inputs/prompt.txt", "r") as f:
                game_state = f.read()
        except:
            pass
        
        # Add moves in PGN format: "1. e4 e5 2. Nf3 Nc6"
        for i, move in enumerate(pgn_moves):
            if i % 2 == 0:  # White move
                move_number = (i // 2) + 1
                if i > 0:  # Add space before move number (except first)
                    game_state += " "
                game_state += f"{move_number}. {move}"
            else:  # Black move
                game_state += f" {move}"
        
        return game_state

def main():
    print("🤖 ChessGPT Lichess Bot")
    print("=" * 40)
    
    # Get token from user
    token = input("Enter your Lichess API token: ").strip()
    if not token:
        print("❌ No token provided")
        return
    
    # Ask about model
    model_name = input("Enter model name (default: large-16-ckpt.pt): ").strip()
    if not model_name:
        model_name = "large-16-ckpt.pt"
    
    # Ask about bot upgrade
    print("\n⚠️  IMPORTANT: This will upgrade your account to a bot account.")
    print("   This is IRREVERSIBLE - you won't be able to play as a human anymore.")
    print("   Make sure you're using a dedicated bot account!")
    
    upgrade = input("\nUpgrade account to bot? (yes/no): ").strip().lower()
    
    if upgrade == 'yes':
        # Create bot instance
        bot = ChessGPTBot(token, model_name)
        
        # Upgrade to bot
        print("\n🔄 Upgrading account to bot...")
        success, response = bot.api.upgrade_to_bot()
        
        if success:
            print("✅ Account upgraded to bot successfully!")
            print("\n🚀 Starting bot...")
            bot.start()
        else:
            print(f"❌ Failed to upgrade account: {response}")
    else:
        print("❌ Bot upgrade cancelled")

if __name__ == "__main__":
    main() 