#!/bin/bash
# Install Stockfish locally for batch processing

echo "Installing Stockfish locally..."

# Create local bin directory
mkdir -p ~/bin
cd ~/bin

# Download Stockfish (latest stable version)
echo "Downloading Stockfish..."
wget -q https://github.com/official-stockfish/Stockfish/releases/download/sf_16/stockfish-ubuntu-x86-64-avx2.tar

if [ $? -ne 0 ]; then
    echo "❌ Download failed. Trying alternative method..."
    
    # Try with curl if wget fails
    curl -L -o stockfish-ubuntu-x86-64-avx2.tar https://github.com/official-stockfish/Stockfish/releases/download/sf_16/stockfish-ubuntu-x86-64-avx2.tar
    
    if [ $? -ne 0 ]; then
        echo "❌ Both wget and curl failed. Please download manually from:"
        echo "https://stockfishchess.org/download/"
        exit 1
    fi
fi

# Extract and set up
echo "Extracting Stockfish..."
tar -xf stockfish-ubuntu-x86-64-avx2.tar
mv stockfish/stockfish-ubuntu-x86-64-avx2 stockfish
chmod +x stockfish

# Clean up
rm -rf stockfish-ubuntu-x86-64-avx2.tar stockfish/

# Add to PATH if not already there
if ! grep -q "~/bin" ~/.bashrc; then
    echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
    echo "Added ~/bin to PATH in ~/.bashrc"
fi

# Test installation
export PATH="$HOME/bin:$PATH"
if command -v stockfish >/dev/null 2>&1; then
    echo "✅ Stockfish installed successfully!"
    echo "Location: $(which stockfish)"
    
    # Quick test
    echo "Testing Stockfish..."
    echo "position startpos" | stockfish | head -5
    
    echo ""
    echo "✅ Installation complete! You can now run:"
    echo "   python test_stockfish_setup.py"
    echo "   python batch_stockfish_processor.py --sample 100"
    
else
    echo "❌ Installation failed. Please try manual installation:"
    echo "1. Download from https://stockfishchess.org/download/"
    echo "2. Extract to ~/bin/stockfish"
    echo "3. Make executable: chmod +x ~/bin/stockfish"
    echo "4. Add to PATH: export PATH=\"\$HOME/bin:\$PATH\""
fi 