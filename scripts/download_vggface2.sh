#!/bin/bash
# Download VGGFace2 dataset
# Note: VGGFace2 requires academic agreement from http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/

set -e

echo "=================================="
echo "VGGFace2 Dataset Download Script"
echo "=================================="

# Configuration
DATA_DIR="${1:-../data}"
mkdir -p "$DATA_DIR"

# Check for required tools
if ! command -v wget &> /dev/null; then
    echo "Error: wget is required. Install with: apt-get install wget"
    exit 1
fi

if ! command -v unzip &> /dev/null; then
    echo "Error: unzip is required. Install with: apt-get install unzip"
    exit 1
fi

echo ""
echo "IMPORTANT: VGGFace2 requires academic agreement."
echo "Please visit: http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/"
echo "to request access and download the dataset."
echo ""
echo "Expected files after download:"
echo "  - vggface2_train.tar.gz (~36GB)"
echo "  - vggface2_test.tar.gz (~2GB)"
echo ""

# Alternative: Download LFW for testing
download_lfw() {
    echo "Downloading LFW dataset for evaluation..."
    LFW_DIR="$DATA_DIR/lfw"
    mkdir -p "$LFW_DIR"
    
    # Download LFW
    if [ ! -f "$LFW_DIR/lfw.tgz" ]; then
        wget -O "$LFW_DIR/lfw.tgz" \
            "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    fi
    
    # Extract
    if [ ! -d "$LFW_DIR/lfw" ]; then
        tar -xzf "$LFW_DIR/lfw.tgz" -C "$LFW_DIR"
    fi
    
    # Download pairs.txt
    if [ ! -f "$DATA_DIR/pairs.txt" ]; then
        wget -O "$DATA_DIR/pairs.txt" \
            "http://vis-www.cs.umass.edu/lfw/pairs.txt"
    fi
    
    echo "LFW downloaded to $LFW_DIR"
}

# Download CFP-FP for evaluation
download_cfp() {
    echo "Downloading CFP-FP dataset for evaluation..."
    CFP_DIR="$DATA_DIR/cfp_fp"
    mkdir -p "$CFP_DIR"
    
    echo "CFP-FP requires manual download from:"
    echo "  http://www.cfpw.io/"
    echo ""
}

# Process function
process_vggface2() {
    echo "Processing VGGFace2 dataset..."
    
    TRAIN_TAR="$DATA_DIR/vggface2_train.tar.gz"
    TEST_TAR="$DATA_DIR/vggface2_test.tar.gz"
    
    if [ -f "$TRAIN_TAR" ]; then
        echo "Extracting training set..."
        mkdir -p "$DATA_DIR/vggface2/train"
        tar -xzf "$TRAIN_TAR" -C "$DATA_DIR/vggface2/"
        echo "Training set extracted."
    else
        echo "Training archive not found: $TRAIN_TAR"
    fi
    
    if [ -f "$TEST_TAR" ]; then
        echo "Extracting test set..."
        mkdir -p "$DATA_DIR/vggface2/test"
        tar -xzf "$TEST_TAR" -C "$DATA_DIR/vggface2/"
        echo "Test set extracted."
    else
        echo "Test archive not found: $TEST_TAR"
    fi
}

# Main menu
echo "Options:"
echo "  1) Download LFW (for evaluation)"
echo "  2) Process VGGFace2 (if already downloaded)"
echo "  3) Both"
echo ""
read -p "Select option [1-3]: " option

case $option in
    1)
        download_lfw
        ;;
    2)
        process_vggface2
        ;;
    3)
        download_lfw
        process_vggface2
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo ""
echo "=================================="
echo "Download complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Run face alignment on the dataset"
echo "2. Update notebook paths"
echo "3. Start training"
