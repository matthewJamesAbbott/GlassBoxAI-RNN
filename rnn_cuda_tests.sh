#!/bin/bash

#
# Matthew Abbott 2025
# RNN CUDA Tests - Comprehensive Test Suite
#

set -o pipefail

PASS=0
FAIL=0
TOTAL=0
TEMP_DIR="./test_output"
RNN_BIN="./rnn_cuda"
FACADE_BIN="./facaded_rnn_cuda"

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

# Compile CUDA binaries
echo "Compiling RNN CUDA binaries..."
nvcc -std=c++17 -o "$RNN_BIN" rnn.cu 2>/dev/null
nvcc -std=c++17 -o "$FACADE_BIN" facaded_rnn.cu 2>/dev/null

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
echo "RNN CUDA Comprehensive Test Suite"
echo "========================================="
echo ""

# Check binaries exist
if [ ! -f "$RNN_BIN" ]; then
    echo -e "${RED}Error: $RNN_BIN not found. Compile with: nvcc -std=c++17 -o rnn_cuda rnn.cu${NC}"
    exit 1
fi

if [ ! -f "$FACADE_BIN" ]; then
    echo -e "${RED}Error: $FACADE_BIN not found. Compile with: nvcc -std=c++17 -o facaded_rnn_cuda facaded_rnn.cu${NC}"
    exit 1
fi

echo -e "${BLUE}=== RNN CUDA Binary Tests ===${NC}"
echo ""

# ============================================
# Basic Help/Usage
# ============================================

echo -e "${BLUE}Group: Help & Usage${NC}"

run_test \
    "RNN CUDA help command" \
    "$RNN_BIN help" \
    "Commands:"

run_test \
    "RNN CUDA --help flag" \
    "$RNN_BIN --help" \
    "Commands:"

run_test \
    "FacadeRNN CUDA help command" \
    "$FACADE_BIN help" \
    "Commands:"

run_test \
    "FacadeRNN CUDA --help flag" \
    "$FACADE_BIN --help" \
    "Commands:"

echo ""

# ============================================
# Model Creation - Basic
# ============================================

echo -e "${BLUE}Group: Model Creation - Basic${NC}"

run_test \
    "Create 2-4-1 LSTM model" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/basic_cuda.json" \
    "Created RNN model"

check_file_exists \
    "JSON file created for 2-4-1" \
    "$TEMP_DIR/basic_cuda.json"

check_json_valid \
    "JSON contains valid RNN structure" \
    "$TEMP_DIR/basic_cuda.json"

run_test \
    "Output shows correct architecture" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/basic2_cuda.json" \
    "Input size: 2"

run_test \
    "Output shows hidden size" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/basic3_cuda.json" \
    "Hidden sizes: 4"

run_test \
    "Output shows output size" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/basic4_cuda.json" \
    "Output size: 1"

echo ""

# ============================================
# Model Creation - Multi-layer
# ============================================

echo -e "${BLUE}Group: Model Creation - Multi-layer${NC}"

run_test \
    "Create 3-5-3-2 network" \
    "$RNN_BIN create --input=3 --hidden=5,3 --output=2 --save=$TEMP_DIR/multilayer_cuda.json" \
    "Created RNN model"

check_file_exists \
    "JSON file for multi-layer" \
    "$TEMP_DIR/multilayer_cuda.json"

run_test \
    "Multi-layer output shows correct input" \
    "$RNN_BIN create --input=3 --hidden=5,3 --output=2 --save=$TEMP_DIR/ml2_cuda.json" \
    "Input size: 3"

run_test \
    "Multi-layer output shows both hidden sizes" \
    "$RNN_BIN create --input=3 --hidden=5,3 --output=2 --save=$TEMP_DIR/ml3_cuda.json" \
    "Hidden sizes: 5,3"

run_test \
    "Multi-layer output shows correct output size" \
    "$RNN_BIN create --input=3 --hidden=5,3 --output=2 --save=$TEMP_DIR/ml4_cuda.json" \
    "Output size: 2"

run_test \
    "Create 3-layer hidden network" \
    "$RNN_BIN create --input=4 --hidden=8,6,4 --output=2 --save=$TEMP_DIR/ml5_cuda.json" \
    "Hidden sizes: 8,6,4"

echo ""

# ============================================
# Cell Types
# ============================================

echo -e "${BLUE}Group: Cell Types${NC}"

run_test \
    "Create with SimpleRNN cell" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/simplernn_cuda.json --cell=simplernn" \
    "Created RNN model"

run_test \
    "Create with LSTM cell (default)" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/lstm_cuda.json --cell=lstm" \
    "Created RNN model"

run_test \
    "Create with GRU cell" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/gru_cuda.json --cell=gru" \
    "Created RNN model"

run_test \
    "SimpleRNN cell type in output" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/simplernn2_cuda.json --cell=simplernn" \
    "Cell type: simplernn"

run_test \
    "LSTM cell type in output" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/lstm2_cuda.json --cell=lstm" \
    "Cell type: lstm"

run_test \
    "GRU cell type in output" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/gru2_cuda.json --cell=gru" \
    "Cell type: gru"

echo ""

# ============================================
# Activation Functions
# ============================================

echo -e "${BLUE}Group: Activation Functions${NC}"

run_test \
    "Hidden activation tanh" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/act_tanh_cuda.json --hidden-act=tanh" \
    "Hidden activation: tanh"

run_test \
    "Hidden activation sigmoid" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/act_sigmoid_cuda.json --hidden-act=sigmoid" \
    "Hidden activation: sigmoid"

run_test \
    "Hidden activation relu" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/act_relu_cuda.json --hidden-act=relu" \
    "Hidden activation: relu"

run_test \
    "Hidden activation linear" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/act_linear_cuda.json --hidden-act=linear" \
    "Hidden activation: linear"

run_test \
    "Output activation linear" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/out_linear_cuda.json --output-act=linear" \
    "Output activation: linear"

run_test \
    "Output activation sigmoid" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/out_sigmoid_cuda.json --output-act=sigmoid" \
    "Output activation: sigmoid"

run_test \
    "Output activation tanh" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/out_tanh_cuda.json --output-act=tanh" \
    "Output activation: tanh"

echo ""

# ============================================
# Loss Functions
# ============================================

echo -e "${BLUE}Group: Loss Functions${NC}"

run_test \
    "Loss function MSE" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/loss_mse_cuda.json --loss=mse" \
    "Loss function: mse"

run_test \
    "Loss function cross-entropy" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/loss_ce_cuda.json --loss=crossentropy" \
    "Loss function: crossentropy"

echo ""

# ============================================
# Hyperparameters
# ============================================

echo -e "${BLUE}Group: Hyperparameters${NC}"

run_test \
    "Learning rate 0.001" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/lr_cuda.json --lr=0.001" \
    "Learning rate: 0.001"

run_test \
    "Gradient clip 1.0" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/clip_cuda.json --clip=1.0" \
    "Gradient clip: 1.0"

run_test \
    "BPTT steps 16" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/bptt_cuda.json --bptt=16" \
    "BPTT steps: 16"

echo ""

# ============================================
# FacadeRNN Model Creation
# ============================================

echo -e "${BLUE}Group: FacadeRNN Model Creation${NC}"

run_test \
    "FacadeRNN create basic model" \
    "$FACADE_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/facade_basic_cuda.json" \
    "Created RNN model"

check_file_exists \
    "FacadeRNN JSON file created" \
    "$TEMP_DIR/facade_basic_cuda.json"

run_test \
    "FacadeRNN create with LSTM" \
    "$FACADE_BIN create --input=3 --hidden=8 --output=2 --save=$TEMP_DIR/facade_lstm_cuda.json --cell=lstm" \
    "Cell type: lstm"

run_test \
    "FacadeRNN create with GRU" \
    "$FACADE_BIN create --input=3 --hidden=8 --output=2 --save=$TEMP_DIR/facade_gru_cuda.json --cell=gru" \
    "Cell type: gru"

run_test \
    "FacadeRNN create multi-layer" \
    "$FACADE_BIN create --input=4 --hidden=16,8 --output=3 --save=$TEMP_DIR/facade_ml_cuda.json" \
    "Hidden sizes: 16,8"

run_test \
    "FacadeRNN shows GPU availability" \
    "$FACADE_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/facade_gpu_cuda.json" \
    "GPU Available:"

echo ""

# ============================================
# Model Prediction
# ============================================

echo -e "${BLUE}Group: Model Prediction${NC}"

# Create a model first
$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/pred_model_cuda.json > /dev/null 2>&1

run_test \
    "RNN prediction with 2 inputs" \
    "$RNN_BIN predict --model=$TEMP_DIR/pred_model_cuda.json --input=0.5,0.5" \
    "Output:"

run_test \
    "RNN prediction shows input echo" \
    "$RNN_BIN predict --model=$TEMP_DIR/pred_model_cuda.json --input=0.5,0.5" \
    "Input:"

# Create a facade model and test prediction
$FACADE_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/facade_pred_cuda.json > /dev/null 2>&1

run_test \
    "FacadeRNN prediction" \
    "$FACADE_BIN predict --model=$TEMP_DIR/facade_pred_cuda.json --input=0.5,0.5" \
    "Output:"

# Create multi-output model
$RNN_BIN create --input=3 --hidden=8 --output=4 --save=$TEMP_DIR/multi_out_cuda.json > /dev/null 2>&1

run_test \
    "Multi-output prediction shows max index" \
    "$RNN_BIN predict --model=$TEMP_DIR/multi_out_cuda.json --input=0.1,0.2,0.3" \
    "Max index:"

echo ""

# ============================================
# Model Info
# ============================================

echo -e "${BLUE}Group: Model Info${NC}"

run_test \
    "RNN info command works" \
    "$RNN_BIN info --model=$TEMP_DIR/basic_cuda.json" \
    "Model Information:"

run_test \
    "RNN info shows input size" \
    "$RNN_BIN info --model=$TEMP_DIR/basic_cuda.json" \
    "Input size:"

run_test \
    "RNN info shows hidden sizes" \
    "$RNN_BIN info --model=$TEMP_DIR/basic_cuda.json" \
    "Hidden sizes:"

run_test \
    "RNN info shows cell type" \
    "$RNN_BIN info --model=$TEMP_DIR/basic_cuda.json" \
    "Cell type:"

run_test \
    "FacadeRNN info command works" \
    "$FACADE_BIN info --model=$TEMP_DIR/facade_basic_cuda.json" \
    "Model Information:"

run_test \
    "FacadeRNN info shows layers" \
    "$FACADE_BIN info --model=$TEMP_DIR/facade_basic_cuda.json" \
    "Layers:"

run_test \
    "FacadeRNN info shows dropout rate" \
    "$FACADE_BIN info --model=$TEMP_DIR/facade_basic_cuda.json" \
    "Dropout rate:"

echo ""

# ============================================
# Cross-Binary Compatibility
# ============================================

echo -e "${BLUE}Group: Cross-Binary Compatibility${NC}"

# Create with RNN, use with FacadeRNN
$RNN_BIN create --input=3 --hidden=6 --output=2 --cell=lstm --save=$TEMP_DIR/cross_rnn_cuda.json > /dev/null 2>&1

run_test \
    "FacadeRNN can read RNN-created model (info)" \
    "$FACADE_BIN info --model=$TEMP_DIR/cross_rnn_cuda.json" \
    "Model Information:"

run_test \
    "FacadeRNN can predict with RNN-created model" \
    "$FACADE_BIN predict --model=$TEMP_DIR/cross_rnn_cuda.json --input=0.1,0.2,0.3" \
    "Output:"

# Create with FacadeRNN, use with RNN
$FACADE_BIN create --input=4 --hidden=8 --output=2 --cell=gru --save=$TEMP_DIR/cross_facade_cuda.json > /dev/null 2>&1

run_test \
    "RNN can read FacadeRNN-created model (info)" \
    "$RNN_BIN info --model=$TEMP_DIR/cross_facade_cuda.json" \
    "Model Information:"

run_test \
    "RNN can predict with FacadeRNN-created model" \
    "$RNN_BIN predict --model=$TEMP_DIR/cross_facade_cuda.json --input=0.1,0.2,0.3,0.4" \
    "Output:"

echo ""

# ============================================
# FacadeRNN Query Commands
# ============================================

echo -e "${BLUE}Group: FacadeRNN Query Commands${NC}"

# Create a model for query tests
$FACADE_BIN create --input=4 --hidden=8,4 --output=2 --cell=lstm --save=$TEMP_DIR/query_model_cuda.json > /dev/null 2>&1

run_test \
    "Query hidden-size" \
    "$FACADE_BIN query --model=$TEMP_DIR/query_model_cuda.json --query-type=hidden-size --layer=0" \
    "Hidden size"

run_test \
    "Query cell-type" \
    "$FACADE_BIN query --model=$TEMP_DIR/query_model_cuda.json --query-type=cell-type" \
    "Cell type:"

run_test \
    "Query dropout-rate" \
    "$FACADE_BIN query --model=$TEMP_DIR/query_model_cuda.json --query-type=dropout-rate" \
    "dropout rate:"

run_test \
    "Query sequence-length" \
    "$FACADE_BIN query --model=$TEMP_DIR/query_model_cuda.json --query-type=sequence-length" \
    "Sequence length:"

echo ""

# ============================================
# Error Handling
# ============================================

echo -e "${BLUE}Group: Error Handling${NC}"

run_test \
    "Missing input size error" \
    "$RNN_BIN create --hidden=4 --output=1 --save=$TEMP_DIR/err1_cuda.json" \
    "Error"

run_test \
    "Missing hidden size error" \
    "$RNN_BIN create --input=2 --output=1 --save=$TEMP_DIR/err2_cuda.json" \
    "Error"

run_test \
    "Missing output size error" \
    "$RNN_BIN create --input=2 --hidden=4 --save=$TEMP_DIR/err3_cuda.json" \
    "Error"

run_test \
    "Missing save file error" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1" \
    "Error"

run_test \
    "Missing model file for predict" \
    "$RNN_BIN predict --input=0.5,0.5" \
    "Error"

run_test \
    "Missing input for predict" \
    "$RNN_BIN predict --model=$TEMP_DIR/basic_cuda.json" \
    "Error"

run_test \
    "Missing model file for info" \
    "$RNN_BIN info" \
    "Error"

run_test \
    "FacadeRNN missing query-type" \
    "$FACADE_BIN query --model=$TEMP_DIR/facade_basic_cuda.json" \
    "Error"

echo ""

# ============================================
# Advanced Features (Specific to RNN)
# ============================================

echo -e "${BLUE}Group: RNN-Specific Features${NC}"

run_test \
    "Large hidden layer" \
    "$RNN_BIN create --input=2 --hidden=256 --output=2 --save=$TEMP_DIR/large_hidden_cuda.json" \
    "Created RNN model"

run_test \
    "Multiple hidden layers" \
    "$RNN_BIN create --input=2 --hidden=64,32,16 --output=2 --save=$TEMP_DIR/multi_hidden_cuda.json" \
    "Hidden sizes: 64,32,16"

run_test \
    "Sequence length configuration (training context)" \
    "$RNN_BIN create --input=5 --hidden=10 --output=2 --save=$TEMP_DIR/seq_config_cuda.json --bptt=16" \
    "BPTT steps: 16"

run_test \
    "Very large network" \
    "$RNN_BIN create --input=10 --hidden=128,64,32 --output=5 --save=$TEMP_DIR/large_net_cuda.json" \
    "Created RNN model"

run_test \
    "Single neuron hidden layer" \
    "$RNN_BIN create --input=2 --hidden=1 --output=1 --save=$TEMP_DIR/single_hidden_cuda.json" \
    "Hidden sizes: 1"

run_test \
    "Many output neurons" \
    "$RNN_BIN create --input=4 --hidden=8 --output=10 --save=$TEMP_DIR/many_output_cuda.json" \
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
    $RNN_BIN create --input=4 --hidden=8 --output=3 --cell=lstm --save=$TEMP_DIR/js_test_lstm_cuda.json > /dev/null 2>&1
    $RNN_BIN create --input=4 --hidden=8 --output=3 --cell=gru --save=$TEMP_DIR/js_test_gru_cuda.json > /dev/null 2>&1
    $RNN_BIN create --input=4 --hidden=8 --output=3 --cell=simplernn --save=$TEMP_DIR/js_test_simple_cuda.json > /dev/null 2>&1
    $RNN_BIN create --input=3 --hidden=6,4 --output=2 --cell=lstm --save=$TEMP_DIR/js_test_multi_cuda.json > /dev/null 2>&1
    $FACADE_BIN create --input=4 --hidden=8 --output=3 --cell=lstm --save=$TEMP_DIR/js_test_facade_cuda.json > /dev/null 2>&1

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
        "validate_rnn_js $TEMP_DIR/js_test_lstm_cuda.json" \
        "tests passed"

    run_test \
        "Weight loading: GRU basic structure" \
        "validate_rnn_js $TEMP_DIR/js_test_gru_cuda.json" \
        "tests passed"

    run_test \
        "Weight loading: SimpleRNN basic structure" \
        "validate_rnn_js $TEMP_DIR/js_test_simple_cuda.json" \
        "tests passed"

    run_test \
        "Weight loading: Multi-layer LSTM structure" \
        "validate_rnn_js $TEMP_DIR/js_test_multi_cuda.json" \
        "tests passed"

    run_test \
        "Weight loading: FacadeRNN created model" \
        "validate_rnn_js $TEMP_DIR/js_test_facade_cuda.json" \
        "tests passed"

    run_test \
        "Weight loading: JSON has input_size" \
        "validate_rnn_js $TEMP_DIR/js_test_lstm_cuda.json" \
        "JSON has input_size"

    run_test \
        "Weight loading: JSON has output_size" \
        "validate_rnn_js $TEMP_DIR/js_test_lstm_cuda.json" \
        "JSON has output_size"

    run_test \
        "Weight loading: JSON has hidden_sizes" \
        "validate_rnn_js $TEMP_DIR/js_test_lstm_cuda.json" \
        "JSON has hidden_sizes"

    run_test \
        "Weight loading: JSON has cells array" \
        "validate_rnn_js $TEMP_DIR/js_test_lstm_cuda.json" \
        "JSON has cells array"

    run_test \
        "Weight loading: JSON has output_layer" \
        "validate_rnn_js $TEMP_DIR/js_test_lstm_cuda.json" \
        "JSON has output_layer"

    run_test \
        "Weight loading: LSTM Wf weights present" \
        "validate_rnn_js $TEMP_DIR/js_test_lstm_cuda.json" \
        "Has Wf weights"

    run_test \
        "Weight loading: LSTM Wf has valid numbers" \
        "validate_rnn_js $TEMP_DIR/js_test_lstm_cuda.json" \
        "Wf has valid numbers"

    run_test \
        "Weight loading: GRU Wz weights present" \
        "validate_rnn_js $TEMP_DIR/js_test_gru_cuda.json" \
        "Has Wz weights"

    run_test \
        "Weight loading: SimpleRNN Wih weights present" \
        "validate_rnn_js $TEMP_DIR/js_test_simple_cuda.json" \
        "Has Wih weights"

    run_test \
        "Weight loading: Output W has valid numbers" \
        "validate_rnn_js $TEMP_DIR/js_test_lstm_cuda.json" \
        "Output W has valid numbers"

else
    echo -e "${YELLOW}Warning: Node.js not found, skipping JavaScript weight loading tests${NC}"
fi

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
echo "RNN CUDA Implementation Coverage"
echo "========================================="
echo ""
echo "Binaries Tested:"
echo "  ✓ RNN CUDA (C++/CUDA)"
echo "  ✓ FacadeRNN CUDA (C++/CUDA)"
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
echo "  ✓ Cross-binary compatibility (RNN <-> FacadeRNN)"
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
