import chess
import chess.svg
from cairosvg import svg2png
import pandas as pd
import os
import io

def load_and_prepare_data(csv_path):
    """Load the chess data from a CSV file and prepare it for visualization."""
    df = pd.read_csv(csv_path)
    # Assuming the CSV contains a 'transcript' column with PGNs
    if 'transcript' not in df.columns:
        raise ValueError("CSV must contain a 'transcript' column with PGN strings.")
    return df

def visualize_game_from_pgn(pgn_string, output_folder="game_visualizations"):
    """
    Generates a sequence of images for a chess game given in PGN format.
    Each image represents a state of the board after a move.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    game = chess.pgn.read_game(io.StringIO(pgn_string))
    if game is None:
        print("Could not read game from PGN string.")
        return

    board = game.board()
    move_number = 0

    # Initial board state
    board_svg = chess.svg.board(board=board)
    output_path = os.path.join(output_folder, f"move_{move_number:03d}.png")
    svg2png(bytestring=board_svg, write_to=output_path)
    print(f"Generated visualization for initial board state: {output_path}")

    for move in game.mainline_moves():
        board.push(move)
        move_number += 1
        board_svg = chess.svg.board(board=board, lastmove=move)
        output_path = os.path.join(output_folder, f"move_{move_number:03d}.png")
        svg2png(bytestring=board_svg, write_to=output_path)
        print(f"Generated visualization for move {move_number}: {output_path}")

def main():
    """
    Main function to drive the visualization.
    It processes a specific game from the dataset.
    """
    # Path to the dataset
    data_path = '../data/processed/lichess_600_T_1100_all_info.csv'
    
    # Load the data
    try:
        chess_data = load_and_prepare_data(data_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

    # Select a game to visualize (e.g., the first game in the dataset)
    if not chess_data.empty:
        # Correctly handle potential leading semicolons or other issues
        pgn_to_visualize = chess_data['transcript'].iloc[0].strip()
        if pgn_to_visualize.startswith(';'):
            pgn_to_visualize = pgn_to_visualize[1:]
            
        print("Visualizing game:")
        print(pgn_to_visualize)
        
        # Define a unique output folder for the selected game
        game_id = chess_data['game_id'].iloc[0] if 'game_id' in chess_data.columns else "game_01"
        output_folder = f"../visualizations/{game_id}_moves"

        visualize_game_from_pgn(pgn_to_visualize, output_folder=output_folder)
        print(f"\nAll visualizations for the selected game have been saved in '{output_folder}'.")
    else:
        print("The dataset is empty or could not be loaded.")

if __name__ == "__main__":
    main() 