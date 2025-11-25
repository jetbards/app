#!/bin/bash

# ICONNET Dashboard Installation Script
# Improved version with modular architecture

echo "🌟 ICONNET Predictive Analytics Dashboard - Installation"
echo "========================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}💡 $1${NC}"
}

# Check if Python is installed
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d " " -f 2)
        print_status "Python found: $PYTHON_VERSION"
        
        # Check if version is 3.7+
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 7 ]; then
            print_status "Python version is compatible"
            return 0
        else
            print_error "Python 3.7+ required. Found: $PYTHON_VERSION"
            return 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.7+"
        return 1
    fi
}

# Check if pip is installed
check_pip() {
    if command -v pip3 &> /dev/null; then
        print_status "pip3 found"
        return 0
    elif command -v pip &> /dev/null; then
        print_status "pip found"
        return 0
    else
        print_error "pip not found. Please install pip"
        return 1
    fi
}

# Install dependencies
install_dependencies() {
    print_info "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
        if [ $? -eq 0 ]; then
            print_status "Dependencies installed successfully"
            return 0
        else
            print_error "Failed to install dependencies"
            return 1
        fi
    else
        print_error "requirements.txt not found"
        return 1
    fi
}

# Create necessary directories
create_directories() {
    print_info "Creating necessary directories..."
    
    directories=("SourceData" "models" "logs")
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_status "Created directory: $dir"
        else
            print_status "Directory exists: $dir"
        fi
    done
}

# Test installation
test_installation() {
    print_info "Testing installation..."
    
    python3 -c "
import sys
sys.path.insert(0, '.')

try:
    from data_manager import DataManager
    from models import ChurnPredictionModel
    from visualizations import DashboardVisualizations
    from main import IconnetDashboard
    print('✅ All modules imported successfully')
    
    # Test data generation
    dm = DataManager()
    df = dm.generate_sample_data(10)
    print(f'✅ Sample data generated: {len(df)} customers')
    
    print('🎉 Installation test passed!')
except Exception as e:
    print(f'❌ Installation test failed: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_status "Installation test passed"
        return 0
    else
        print_error "Installation test failed"
        return 1
    fi
}

# Main installation process
main() {
    echo
    print_info "Starting installation process..."
    echo
    
    # Check Python
    if ! check_python; then
        exit 1
    fi
    
    # Check pip
    if ! check_pip; then
        exit 1
    fi
    
    # Install dependencies
    if ! install_dependencies; then
        exit 1
    fi
    
    # Create directories
    create_directories
    
    # Test installation
    if ! test_installation; then
        exit 1
    fi
    
    echo
    print_status "🎉 Installation completed successfully!"
    echo
    print_info "To run the dashboard:"
    echo -e "${BLUE}   python3 run_dashboard.py${NC}"
    echo -e "${BLUE}   # OR${NC}"
    echo -e "${BLUE}   streamlit run main.py${NC}"
    echo
    print_info "Dashboard will be available at: http://localhost:8501"
    echo
    print_info "For more information, read README.md"
    echo
}

# Run main function
main "$@"