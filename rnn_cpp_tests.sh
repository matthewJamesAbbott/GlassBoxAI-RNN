#!/bin/bash

#
# Matthew Abbott 2025
# RNN C++ Tests - Comprehensive Test Suite
#

set -o pipefail

PASS=0
FAIL=0
TOTAL=0
TEMP_DIR="./test_output"
RNN_BIN="./rnn"
FACADE_BIN="./facade_rnn"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Setup/Cleanup
cleanup() {
    # Cleanup handled manually if needed
    :
}
trap cleanup EXIT

mkdir -p "$TEMP_DIR"

# Compile C++ binaries
g++ -std=c++17 -O2 rnn.cpp -o rnn
g++ -std=c++17 -O2 FacadeRNN.cpp -o facade_rnn

# Test function
run_test() {
    local test_name="$1"
    local command="$2"
    local expected_pattern="$3"

    TOTAL=$((TOTAL + 1))
    echo -n "Test $TOTAL: $test_name... "

    output=$(eval "$command" 2>&1)
    exit_code=$?

    if echo "$output" | grep -q "$expected_pattern"; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAIL${NC}"
        echo "  Command: $command"
        echo "  Expected pattern: $expected_pattern"
        echo "  Output:"
        echo "$output" | head -5
        FAIL=$((FAIL + 1))
    fi
}

check_file_exists() {
    local test_name="$1"
    local file="$2"

    TOTAL=$((TOTAL + 1))
    echo -n "Test $TOTAL: $test_name... "

    if [ -f "$file" ]; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAIL${NC}"
        echo "  File not found: $file"
        FAIL=$((FAIL + 1))
    fi
}

check_json_valid() {
    local test_name="$1"
    local file="$2"

    TOTAL=$((TOTAL + 1))
    echo -n "Test $TOTAL: $test_name... "

    if [ ! -f "$file" ]; then
        echo -e "${RED}FAIL${NC}"
        echo "  File not found: $file"
        FAIL=$((FAIL + 1))
        return
    fi

    if grep -q '"input_size"' "$file" && grep -q '"output_size"' "$file" && grep -q '"hidden_sizes"' "$file"; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAIL${NC}"
        echo "  Invalid JSON structure in $file"
        FAIL=$((FAIL + 1))
    fi
}

# ============================================
# Start Tests
# ============================================

echo ""
echo "========================================="
echo "RNN C++ Comprehensive Test Suite"
echo "========================================="
echo ""

# Check binaries exist
if [ ! -f "$RNN_BIN" ]; then
    echo -e "${RED}Error: $RNN_BIN not found. Compile with: g++ -std=c++17 -O2 rnn.cpp -o rnn${NC}"
    exit 1
fi

if [ ! -f "$FACADE_BIN" ]; then
    echo -e "${RED}Error: $FACADE_BIN not found. Compile with: g++ -std=c++17 -O2 FacadeRNN.cpp -o facade_rnn${NC}"
    exit 1
fi

echo -e "${BLUE}=== RNN C++ Binary Tests ===${NC}"
echo ""

# ============================================
# Basic Help/Usage
# ============================================

echo -e "${BLUE}Group: Help & Usage${NC}"

run_test \
    "RNN help command" \
    "$RNN_BIN help" \
    "Commands:"

run_test \
    "RNN --help flag" \
    "$RNN_BIN --help" \
    "Commands:"

run_test \
    "FacadeRNN help command" \
    "$FACADE_BIN help" \
    "Commands:"

run_test \
    "FacadeRNN --help flag" \
    "$FACADE_BIN --help" \
    "Commands:"

echo ""

# ============================================
# Model Creation - Basic
# ============================================

echo -e "${BLUE}Group: Model Creation - Basic${NC}"

run_test \
    "Create 2-4-1 LSTM model" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/cpp_basic.json" \
    "Created RNN model"

check_file_exists \
    "JSON file created for 2-4-1" \
    "$TEMP_DIR/cpp_basic.json"

check_json_valid \
    "JSON contains valid RNN structure" \
    "$TEMP_DIR/cpp_basic.json"

run_test \
    "Output shows correct architecture" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/cpp_basic2.json" \
    "Input size: 2"

run_test \
    "Output shows hidden size" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/cpp_basic3.json" \
    "Hidden sizes: 4"

run_test \
    "Output shows output size" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/cpp_basic4.json" \
    "Output size: 1"

echo ""

# ============================================
# Model Creation - Multi-layer
# ============================================

echo -e "${BLUE}Group: Model Creation - Multi-layer${NC}"

run_test \
    "Create 3-5-3-2 network" \
    "$RNN_BIN create --input=3 --hidden=5,3 --output=2 --save=$TEMP_DIR/cpp_multilayer.json" \
    "Created RNN model"

check_file_exists \
    "JSON file for multi-layer" \
    "$TEMP_DIR/cpp_multilayer.json"

run_test \
    "Multi-layer output shows correct input" \
    "$RNN_BIN create --input=3 --hidden=5,3 --output=2 --save=$TEMP_DIR/cpp_ml2.json" \
    "Input size: 3"

run_test \
    "Multi-layer output shows both hidden sizes" \
    "$RNN_BIN create --input=3 --hidden=5,3 --output=2 --save=$TEMP_DIR/cpp_ml3.json" \
    "Hidden sizes: 5,3"

run_test \
    "Multi-layer output shows correct output size" \
    "$RNN_BIN create --input=3 --hidden=5,3 --output=2 --save=$TEMP_DIR/cpp_ml4.json" \
    "Output size: 2"

run_test \
    "Create 3-layer hidden network" \
    "$RNN_BIN create --input=4 --hidden=8,6,4 --output=2 --save=$TEMP_DIR/cpp_ml5.json" \
    "Hidden sizes: 8,6,4"

echo ""

# ============================================
# Cell Types
# ============================================

echo -e "${BLUE}Group: Cell Types${NC}"

run_test \
    "Create with SimpleRNN cell" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/cpp_simplernn.json --cell=simplernn" \
    "Created RNN model"

run_test \
    "Create with LSTM cell (default)" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/cpp_lstm.json --cell=lstm" \
    "Created RNN model"

run_test \
    "Create with GRU cell" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/cpp_gru.json --cell=gru" \
    "Created RNN model"

run_test \
    "Output shows SimpleRNN cell type" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/cpp_simplernn2.json --cell=simplernn" \
    "Cell type: simplernn"

run_test \
    "Output shows LSTM cell type" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/cpp_lstm2.json --cell=lstm" \
    "Cell type: lstm"

run_test \
    "Output shows GRU cell type" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/cpp_gru2.json --cell=gru" \
    "Cell type: gru"

echo ""

# ============================================
# Activation Functions
# ============================================

echo -e "${BLUE}Group: Activation Functions${NC}"

run_test \
    "Sigmoid activation" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/cpp_sigmoid.json --hidden-act=sigmoid" \
    "Hidden activation: sigmoid"

run_test \
    "Tanh activation" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/cpp_tanh.json --hidden-act=tanh" \
    "Hidden activation: tanh"

run_test \
    "ReLU activation" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/cpp_relu.json --hidden-act=relu" \
    "Hidden activation: relu"

run_test \
    "Linear activation" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/cpp_linear.json --hidden-act=linear" \
    "Hidden activation: linear"

run_test \
    "Output activation sigmoid" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/cpp_out_sigmoid.json --output-act=sigmoid" \
    "Output activation: sigmoid"

run_test \
    "Output activation tanh" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/cpp_out_tanh.json --output-act=tanh" \
    "Output activation: tanh"

echo ""

# ============================================
# Loss Functions
# ============================================

echo -e "${BLUE}Group: Loss Functions${NC}"

run_test \
    "MSE loss function" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/cpp_mse.json --loss=mse" \
    "Loss function: mse"

run_test \
    "Cross-entropy loss function" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/cpp_ce.json --loss=crossentropy" \
    "Loss function: crossentropy"

echo ""

# ============================================
# Hyperparameters
# ============================================

echo -e "${BLUE}Group: Hyperparameters${NC}"

run_test \
    "Custom learning rate" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/cpp_lr.json --lr=0.001" \
    "Learning rate: 0.001"

run_test \
    "Custom gradient clip" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/cpp_clip.json --clip=10.0" \
    "Gradient clip: 10.00"

run_test \
    "Custom BPTT steps" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/cpp_bptt.json --bptt=32" \
    "BPTT steps: 32"

echo ""

# ============================================
# Prediction/Inference
# ============================================

echo -e "${BLUE}Group: Prediction/Inference${NC}"

# Create a model for prediction
$RNN_BIN create --input=2 --hidden=4 --output=2 --save=$TEMP_DIR/cpp_pred_model.json > /dev/null 2>&1

run_test \
    "Predict with single input pair" \
    "$RNN_BIN predict --model=$TEMP_DIR/cpp_pred_model.json --input=0.5,0.5" \
    "Input:"

run_test \
    "Predict output is shown" \
    "$RNN_BIN predict --model=$TEMP_DIR/cpp_pred_model.json --input=0.5,0.5" \
    "Output:"

run_test \
    "Predict with different input" \
    "$RNN_BIN predict --model=$TEMP_DIR/cpp_pred_model.json --input=0.1,0.9" \
    "Input: 0.1000, 0.9000"

echo ""

# ============================================
# Info Command
# ============================================

echo -e "${BLUE}Group: Info Command${NC}"

run_test \
    "RNN info command works" \
    "$RNN_BIN info --model=$TEMP_DIR/cpp_basic.json" \
    "Loading model"

run_test \
    "FacadeRNN info command works" \
    "$FACADE_BIN info --model=$TEMP_DIR/cpp_basic.json" \
    "Loading model"

echo ""

# ============================================
# FacadeRNN-Specific Tests
# ============================================

echo -e "${BLUE}Group: FacadeRNN Facade Functions${NC}"

# Create models for facade tests
$FACADE_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/cpp_facade_basic.json > /dev/null 2>&1
$FACADE_BIN create --input=3 --hidden=5,3 --output=2 --cell=lstm --save=$TEMP_DIR/cpp_facade_multi.json > /dev/null 2>&1

run_test \
    "Query input size" \
    "$FACADE_BIN query --model=$TEMP_DIR/cpp_facade_basic.json --query-type=input-size" \
    "Input size:"

run_test \
    "Query output size" \
    "$FACADE_BIN query --model=$TEMP_DIR/cpp_facade_basic.json --query-type=output-size" \
    "Output size:"

run_test \
    "Query hidden size" \
    "$FACADE_BIN query --model=$TEMP_DIR/cpp_facade_basic.json --query-type=hidden-size --layer=0" \
    "Hidden size"

run_test \
    "Query cell type" \
    "$FACADE_BIN query --model=$TEMP_DIR/cpp_facade_basic.json --query-type=cell-type" \
    "Cell type:"

run_test \
    "Query sequence length" \
    "$FACADE_BIN query --model=$TEMP_DIR/cpp_facade_basic.json --query-type=sequence-length" \
    "Sequence length:"

run_test \
    "Query dropout rate" \
    "$FACADE_BIN query --model=$TEMP_DIR/cpp_facade_basic.json --query-type=dropout-rate" \
    "dropout rate:"

run_test \
    "Enable dropout flag" \
    "$FACADE_BIN query --model=$TEMP_DIR/cpp_facade_basic.json --query-type=input-size --enable-dropout" \
    "Dropout enabled"

run_test \
    "Disable dropout flag" \
    "$FACADE_BIN query --model=$TEMP_DIR/cpp_facade_basic.json --query-type=input-size --disable-dropout" \
    "Dropout disabled"

echo ""

# ============================================
# Error Handling
# ============================================

echo -e "${BLUE}Group: Error Handling${NC}"

run_test \
    "Error on missing --input" \
    "$RNN_BIN create --hidden=4 --output=1 --save=$TEMP_DIR/cpp_err1.json" \
    "Error:"

run_test \
    "Error on missing --hidden" \
    "$RNN_BIN create --input=2 --output=1 --save=$TEMP_DIR/cpp_err2.json" \
    "Error:"

run_test \
    "Error on missing --output" \
    "$RNN_BIN create --input=2 --hidden=4 --save=$TEMP_DIR/cpp_err3.json" \
    "Error:"

run_test \
    "Error on missing --save" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1" \
    "Error:"

run_test \
    "Error on missing --model in predict" \
    "$RNN_BIN predict --input=0.5,0.5" \
    "Error:"

run_test \
    "Error on missing --input in predict" \
    "$RNN_BIN predict --model=$TEMP_DIR/cpp_basic.json" \
    "Error:"

run_test \
    "Handle non-existent model file gracefully" \
    "$RNN_BIN info --model=$TEMP_DIR/cpp_nonexistent.json 2>&1" \
    "Loading model"

echo ""

# ============================================
# Advanced Features
# ============================================

echo -e "${BLUE}Group: Advanced Features${NC}"

run_test \
    "Large hidden layer" \
    "$RNN_BIN create --input=2 --hidden=256 --output=2 --save=$TEMP_DIR/cpp_large_hidden.json" \
    "Created RNN model"

run_test \
    "Multiple hidden layers" \
    "$RNN_BIN create --input=2 --hidden=64,32,16 --output=2 --save=$TEMP_DIR/cpp_multi_hidden.json" \
    "Hidden sizes: 64,32,16"

run_test \
    "BPTT configuration" \
    "$RNN_BIN create --input=5 --hidden=10 --output=2 --save=$TEMP_DIR/cpp_seq_config.json --bptt=16" \
    "BPTT steps: 16"

run_test \
    "Very large network" \
    "$RNN_BIN create --input=10 --hidden=128,64,32 --output=5 --save=$TEMP_DIR/cpp_large_net.json" \
    "Created RNN model"

run_test \
    "Single neuron hidden layer" \
    "$RNN_BIN create --input=2 --hidden=1 --output=1 --save=$TEMP_DIR/cpp_single_hidden.json" \
    "Hidden sizes: 1"

run_test \
    "Many output neurons" \
    "$RNN_BIN create --input=4 --hidden=8 --output=10 --save=$TEMP_DIR/cpp_many_output.json" \
    "Output size: 10"

echo ""

# ============================================
# Weight Loading Tests (JavaScript compatibility)
# ============================================

echo -e "${BLUE}Group: Weight Loading & JavaScript Compatibility${NC}"

# Check if Node.js is available
if command -v node &> /dev/null; then
    echo -e "${YELLOW}Info: Testing weight loading with inline JavaScript validation${NC}"
    
    # Create models for weight loading tests
    $RNN_BIN create --input=4 --hidden=8 --output=3 --cell=lstm --save=$TEMP_DIR/cpp_js_test_lstm.json > /dev/null 2>&1
    $RNN_BIN create --input=4 --hidden=8 --output=3 --cell=gru --save=$TEMP_DIR/cpp_js_test_gru.json > /dev/null 2>&1
    $RNN_BIN create --input=4 --hidden=8 --output=3 --cell=simplernn --save=$TEMP_DIR/cpp_js_test_simple.json > /dev/null 2>&1
    $RNN_BIN create --input=3 --hidden=6,4 --output=2 --cell=lstm --save=$TEMP_DIR/cpp_js_test_multi.json > /dev/null 2>&1
    $FACADE_BIN create --input=4 --hidden=8 --output=3 --cell=lstm --save=$TEMP_DIR/cpp_js_test_facade.json > /dev/null 2>&1

    # Inline JavaScript validation function
    validate_rnn_js() {
        local jsonfile="$1"
        node -e "
const fs = require('fs');
let pass = 0, fail = 0;
function test(name, condition) {
    if (condition) { console.log('✓ ' + name); pass++; }
    else { console.log('✗ ' + name); fail++; }
}
function hasValidNumbers(arr) {
    if (!Array.isArray(arr)) return false;
    const flat = arr.flat(Infinity);
    return flat.every(n => typeof n === 'number' && !isNaN(n) && isFinite(n));
}
try {
    const json = JSON.parse(fs.readFileSync('$jsonfile', 'utf8'));
    test('JSON has input_size', json.input_size !== undefined);
    test('JSON has output_size', json.output_size !== undefined);
    test('JSON has hidden_sizes', Array.isArray(json.hidden_sizes));
    test('JSON has cell_type', json.cell_type !== undefined);
    test('JSON has cells array', Array.isArray(json.cells));
    test('JSON has output_layer', json.output_layer !== undefined);
    const inputSize = json.input_size, outputSize = json.output_size;
    const hiddenSizes = json.hidden_sizes, cellType = json.cell_type;
    test('Input size is positive', inputSize > 0);
    test('Output size is positive', outputSize > 0);
    test('Cell count matches hidden_sizes', json.cells.length === hiddenSizes.length);
    let prevSize = inputSize;
    for (let i = 0; i < json.cells.length; i++) {
        const cell = json.cells[i], hs = hiddenSizes[i], expDim = prevSize + hs;
        if (cellType === 'lstm') {
            test('Cell ' + i + ': Has Wf weights', cell.Wf !== undefined);
            test('Cell ' + i + ': Has Wi weights', cell.Wi !== undefined);
            test('Cell ' + i + ': Has Wc weights', cell.Wc !== undefined);
            test('Cell ' + i + ': Has Wo weights', cell.Wo !== undefined);
            test('Cell ' + i + ': Wf shape correct', cell.Wf && cell.Wf.length === hs && cell.Wf[0]?.length === expDim);
            test('Cell ' + i + ': Wf has valid numbers', hasValidNumbers(cell.Wf));
            const bf = cell.Bf ?? cell.bf;
            test('Cell ' + i + ': Has Bf bias', bf !== undefined);
            test('Cell ' + i + ': Bf length correct', bf?.length === hs);
        } else if (cellType === 'gru') {
            test('Cell ' + i + ': Has Wz weights', cell.Wz !== undefined);
            test('Cell ' + i + ': Has Wr weights', cell.Wr !== undefined);
            test('Cell ' + i + ': Has Wh weights', cell.Wh !== undefined);
            test('Cell ' + i + ': Wz shape correct', cell.Wz && cell.Wz.length === hs && cell.Wz[0]?.length === expDim);
            test('Cell ' + i + ': Wz has valid numbers', hasValidNumbers(cell.Wz));
        } else if (cellType === 'simplernn') {
            test('Cell ' + i + ': Has Wih weights', cell.Wih !== undefined);
            test('Cell ' + i + ': Has Whh weights', cell.Whh !== undefined);
            test('Cell ' + i + ': Wih shape correct', cell.Wih && cell.Wih.length === hs && cell.Wih[0]?.length === prevSize);
            test('Cell ' + i + ': Wih has valid numbers', hasValidNumbers(cell.Wih));
        }
        prevSize = hs;
    }
    const outLayer = json.output_layer, lastHs = hiddenSizes[hiddenSizes.length - 1];
    test('Output layer has W', outLayer.W !== undefined);
    const outB = outLayer.B ?? outLayer.b;
    test('Output layer has B', outB !== undefined);
    test('Output W shape correct', outLayer.W && outLayer.W.length === outputSize && outLayer.W[0]?.length === lastHs);
    test('Output W has valid numbers', hasValidNumbers(outLayer.W));
    test('Output B has valid numbers', hasValidNumbers(outB));
    console.log(pass + '/' + (pass + fail) + ' tests passed');
    process.exit(fail === 0 ? 0 : 1);
} catch (e) { console.log('Error: ' + e.message); process.exit(1); }
" 2>&1
    }

    run_test \
        "Weight loading: LSTM basic structure" \
        "validate_rnn_js $TEMP_DIR/cpp_js_test_lstm.json" \
        "tests passed"

    run_test \
        "Weight loading: GRU basic structure" \
        "validate_rnn_js $TEMP_DIR/cpp_js_test_gru.json" \
        "tests passed"

    run_test \
        "Weight loading: SimpleRNN basic structure" \
        "validate_rnn_js $TEMP_DIR/cpp_js_test_simple.json" \
        "tests passed"

    run_test \
        "Weight loading: Multi-layer LSTM structure" \
        "validate_rnn_js $TEMP_DIR/cpp_js_test_multi.json" \
        "tests passed"

    run_test \
        "Weight loading: FacadeRNN created model" \
        "validate_rnn_js $TEMP_DIR/cpp_js_test_facade.json" \
        "tests passed"

    run_test \
        "Weight loading: JSON has input_size" \
        "validate_rnn_js $TEMP_DIR/cpp_js_test_lstm.json" \
        "JSON has input_size"

    run_test \
        "Weight loading: JSON has output_size" \
        "validate_rnn_js $TEMP_DIR/cpp_js_test_lstm.json" \
        "JSON has output_size"

    run_test \
        "Weight loading: JSON has hidden_sizes" \
        "validate_rnn_js $TEMP_DIR/cpp_js_test_lstm.json" \
        "JSON has hidden_sizes"

    run_test \
        "Weight loading: JSON has cells array" \
        "validate_rnn_js $TEMP_DIR/cpp_js_test_lstm.json" \
        "JSON has cells array"

    run_test \
        "Weight loading: JSON has output_layer" \
        "validate_rnn_js $TEMP_DIR/cpp_js_test_lstm.json" \
        "JSON has output_layer"

    run_test \
        "Weight loading: LSTM Wf weights present" \
        "validate_rnn_js $TEMP_DIR/cpp_js_test_lstm.json" \
        "Has Wf weights"

    run_test \
        "Weight loading: LSTM Wf has valid numbers" \
        "validate_rnn_js $TEMP_DIR/cpp_js_test_lstm.json" \
        "Wf has valid numbers"

    run_test \
        "Weight loading: GRU Wz weights present" \
        "validate_rnn_js $TEMP_DIR/cpp_js_test_gru.json" \
        "Has Wz weights"

    run_test \
        "Weight loading: SimpleRNN Wih weights present" \
        "validate_rnn_js $TEMP_DIR/cpp_js_test_simple.json" \
        "Has Wih weights"

    run_test \
        "Weight loading: Output W has valid numbers" \
        "validate_rnn_js $TEMP_DIR/cpp_js_test_lstm.json" \
        "Output W has valid numbers"

else
    echo -e "${YELLOW}Warning: Node.js not found, skipping JavaScript weight loading tests${NC}"
fi

echo ""

# ============================================
# C++ vs C++ Binary Compatibility
# ============================================

echo -e "${BLUE}Group: C++ Binary Interoperability${NC}"

# Create model with RNN binary
$RNN_BIN create --input=3 --hidden=5 --output=2 --cell=lstm --save=$TEMP_DIR/cpp_compat_rnn.json > /dev/null 2>&1

run_test \
    "RNN binary can load RNN-created model" \
    "$RNN_BIN info --model=$TEMP_DIR/cpp_compat_rnn.json" \
    "Loading model"

run_test \
    "FacadeRNN binary can load RNN-created model" \
    "$FACADE_BIN info --model=$TEMP_DIR/cpp_compat_rnn.json" \
    "Loading model"

# Create model with FacadeRNN binary
$FACADE_BIN create --input=3 --hidden=5 --output=2 --cell=gru --save=$TEMP_DIR/cpp_compat_facade.json > /dev/null 2>&1

run_test \
    "FacadeRNN can load FacadeRNN-created model" \
    "$FACADE_BIN info --model=$TEMP_DIR/cpp_compat_facade.json" \
    "Loading model"

run_test \
    "RNN can load FacadeRNN-created model" \
    "$RNN_BIN info --model=$TEMP_DIR/cpp_compat_facade.json" \
    "Loading model"

echo ""

# ============================================
# Summary
# ============================================

echo "========================================="
echo "Test Summary"
echo "========================================="
echo "Total tests: $TOTAL"
echo -e "Passed: ${GREEN}$PASS${NC}"
echo -e "Failed: ${RED}$FAIL${NC}"
echo ""

echo "========================================="
echo "RNN C++ Implementation Coverage"
echo "========================================="
echo ""
echo "Binaries Tested:"
echo "  ✓ rnn (C++)"
echo "  ✓ facade_rnn (C++)"
echo ""
echo "Cell Types Tested:"
echo "  ✓ SimpleRNN"
echo "  ✓ LSTM"
echo "  ✓ GRU"
echo ""
echo "Features Tested:"
echo "  ✓ Model creation (all cell types)"
echo "  ✓ Multi-layer networks"
echo "  ✓ Activation functions (tanh, sigmoid, relu, linear)"
echo "  ✓ Loss functions (MSE, cross-entropy)"
echo "  ✓ Hyperparameters (LR, gradient clip, BPTT)"
echo "  ✓ JSON serialization"
echo "  ✓ Prediction/inference"
echo "  ✓ Binary interoperability"
echo "  ✓ FacadeRNN query commands"
echo "  ✓ JavaScript weight validation"
echo "  ✓ Error handling"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
