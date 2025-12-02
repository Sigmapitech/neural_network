#!/usr/bin/env bash
# Quick-start demo for MY_TORCH Chess Analyzer

set -e

# Cleanup option
if [ "$1" = "clean" ]; then
    echo "Cleaning demo artifacts..."
    rm -f data/demo_dataset.txt data/demo_test.txt
    rm -f demo_network.nn demo_network_trained.nn
    rm -f basic_chess_network_*.nn
    echo "✓ Cleanup complete!"
    exit 0
fi

echo "=== MY_TORCH Chess Analyzer - Quick Start ==="
echo ""

VENV=venv

if command -v python3 &>/dev/null; then
  PYTHON=python3
elif command -v python &>/dev/null; then
  PYTHON=python
else
  echo "No python or python3 executable found." >&2
  exit 1
fi

# Bootstrap environment if needed
if [ ! -d "$VENV" ] && [ ! -f "my_torch_analyzer" ]; then
    echo "0. Setting up environment (first run only)..."
    if [ -f "/.dockerenv" ]; then
        ./bootstrap-docker-venv.sh
        $VENV/bin/pip install -e .
    else
        python3 -m venv $VENV
        $VENV/bin/pip install -e .
    fi
    cat > my_torch_analyzer << 'WRAPPER'
#!/usr/bin/env bash
exec "$VENV/bin/$PYTHON" "$(dirname "$0")/src/my_torch_analyzer.py" "$@"
WRAPPER
    cat > my_torch_generator << 'WRAPPER'
#!/usr/bin/env bash
exec "$(dirname "$0")/$VENV/bin/$PYTHON" "$(dirname "$0")/src/my_torch_generator.py" "$@"
WRAPPER
    chmod +x my_torch_analyzer my_torch_generator
    echo "✓ Environment ready!"
    echo ""
fi

echo "1. Generating sample dataset (500 positions)..."
$VENV/bin/$PYTHON scripts/generate_chess_dataset.py sample data/demo_dataset.txt --count 500

echo ""
echo "2. Showing dataset statistics..."
$VENV/bin/$PYTHON scripts/generate_chess_dataset.py stats data/demo_dataset.txt

echo ""
echo "3. Generating a custom network config for demo..."
cat > /tmp/demo_network.conf << 'NETCONF'
{
  "name": "demo_chess_network",
  "layer_sizes": [64, 128, 64, 5],
  "hidden_activation": "relu",
  "output_activation": "softmax",
  "loss": "ce",
  "optimizer": {
    "type": "sgd",
    "learning_rate": 0.01,
    "momentum": 0.9,
    "lr_decay": 0.0
  },
  "seed": 42
}
NETCONF
./my_torch_generator /tmp/demo_network.conf 1
mv demo_chess_network_1.nn demo_network.nn

echo ""
echo "4. Training network (this may take 1-2 minutes)..."
./my_torch_analyzer --train demo_network.nn data/demo_dataset.txt --save demo_network_trained.nn --epochs 200

echo ""
echo "5. Creating test file..."
cat > data/demo_test.txt << 'EOF'
rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3 Checkmate White
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 Nothing
rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3 Check Black
EOF

echo ""
echo "6. Running predictions..."
echo "Input positions:"
cat data/demo_test.txt
echo ""
echo "Predictions:"
./my_torch_analyzer --predict demo_network_trained.nn data/demo_test.txt

echo ""
echo "=== Demo Complete! ==="
echo ""
echo "Try these commands:"
echo "  ./my_torch_analyzer --help"
echo "  ./my_torch_generator --help"
echo "  python3 scripts/generate_chess_dataset.py --help"
echo ""
echo "Generated files:"
echo "  - data/demo_dataset.txt (training data)"
echo "  - data/demo_test.txt (test positions)"
echo "  - demo_network.nn (initial network)"
echo "  - demo_network_trained.nn (trained network)"
echo ""
echo "To clean up demo files: ./demo.sh clean"
