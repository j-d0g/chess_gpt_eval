#!/bin/bash
#
# Launch the interactive Stockfish analysis dashboard
#

echo "Starting Chess Model Stockfish Analysis Dashboard..."
echo "The dashboard will be available at: http://localhost:8050"
echo ""
echo "If you're running this on a remote server, you may need to:"
echo "1. Use SSH port forwarding: ssh -L 8050:localhost:8050 username@server"
echo "2. Or modify the script to use a different host/port"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

python -c "from src.visualization.interactive_dashboard import AdvancedChessAnalysisDashboard; dashboard = AdvancedChessAnalysisDashboard(); dashboard.run()" 