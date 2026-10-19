#!/usr/bin/env bash
# ==============================================================================
#  Psyduck - Server Manager & Automated Setup (CLI + TUI)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Formatting
BOLD='\033[1m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RESET='\033[0m'

# Direct Dependencies (Unpinned without version locks)
DIRECT_DEPENDENCIES=(
    "fastapi"
    "uvicorn"
    "jinja2"
    "requests"
    "python-dotenv"
    "sentence-transformers"
    "faiss-cpu"
    "numpy"
    "pandas"
    "kagglehub"
)

VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"
DATA_DIR="$SCRIPT_DIR/data"
CSV_FILE="$DATA_DIR/data.csv"
DB_FILE="$DATA_DIR/fashion.db"
INDEX_FILE="$DATA_DIR/fashion.index"
IMAGES_DIR="$DATA_DIR/database"
OLLAMA_SCRIPT="$SCRIPT_DIR/scripts/ollama_setup.sh"
ENV_FILE="$SCRIPT_DIR/.env"

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------
print_banner() {
    echo -e "${BOLD}${CYAN}"
    cat << "EOF"
  ============================================================================
    PSYDUCK - Fashion Outfit Assistant Server Manager
  ============================================================================
EOF
    echo -e "${RESET}"
}

has_cmd() {
    command -v "$1" >/dev/null 2>&1
}

detect_pkg_manager() {
    if has_cmd uv; then
        echo "uv"
    elif has_cmd pip3; then
        echo "pip3"
    elif has_cmd pip; then
        echo "pip"
    else
        echo "none"
    fi
}

ensure_venv() {
    if [[ ! -d "$VENV_DIR" || ! -f "$PYTHON_BIN" ]]; then
        echo -e "${YELLOW}[INFO] Creating Python virtual environment in .venv...${RESET}"
        if has_cmd uv; then
            uv venv "$VENV_DIR"
        else
            python3 -m venv "$VENV_DIR"
        fi
        echo -e "${GREEN}[SUCCESS] Virtual environment initialized.${RESET}"
    fi
}

get_ollama_status() {
    if [[ -x "$OLLAMA_SCRIPT" ]]; then
        bash "$OLLAMA_SCRIPT" --status 2>/dev/null || echo "VERSION=UNKNOWN SERVICE=UNKNOWN"
    elif has_cmd ollama; then
        local ver
        ver="$(ollama --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1 || echo 'INSTALLED')"
        echo "VERSION=$ver SERVICE=active"
    else
        echo "VERSION=NOT_INSTALLED SERVICE=inactive"
    fi
}

check_dependencies_installed() {
    if [[ ! -f "$PYTHON_BIN" ]]; then
        return 1
    fi
    "$PYTHON_BIN" -c "import fastapi, uvicorn, jinja2, requests, dotenv, sentence_transformers, faiss, numpy, pandas, kagglehub" >/dev/null 2>&1
}

print_system_status() {
    echo -e "${BOLD}System Status Overview:${RESET}"

    # Virtual environment
    if [[ -f "$PYTHON_BIN" ]]; then
        local py_ver
        py_ver="$("$PYTHON_BIN" --version 2>/dev/null || echo 'Python')"
        echo -e "  - Virtualenv (.venv)    : ${GREEN}[OK]${RESET} ($py_ver)"
    else
        echo -e "  - Virtualenv (.venv)    : ${YELLOW}[NOT INITIALIZED]${RESET}"
    fi

    # Package Manager
    local pm
    pm="$(detect_pkg_manager)"
    if [[ "$pm" == "uv" ]]; then
        echo -e "  - Package Manager       : ${GREEN}[OK]${RESET} uv (high performance)"
    elif [[ "$pm" != "none" ]]; then
        echo -e "  - Package Manager       : ${GREEN}[OK]${RESET} $pm"
    else
        echo -e "  - Package Manager       : ${RED}[ERROR]${RESET} neither uv nor pip available"
    fi

    # Dependencies
    if check_dependencies_installed; then
        echo -e "  - Direct Dependencies   : ${GREEN}[OK]${RESET} All packages installed"
    else
        echo -e "  - Direct Dependencies   : ${YELLOW}[PENDING]${RESET} Installation required"
    fi

    # Dataset
    local img_cnt=0
    if [[ -d "$IMAGES_DIR" ]]; then
        img_cnt="$(ls -1 "$IMAGES_DIR" 2>/dev/null | wc -l)"
    fi
    if [[ -f "$CSV_FILE" && $img_cnt -gt 0 ]]; then
        echo -e "  - Fashion Dataset       : ${GREEN}[OK]${RESET} data.csv ($img_cnt images in database/)"
    elif [[ -f "$CSV_FILE" ]]; then
        echo -e "  - Fashion Dataset       : ${YELLOW}[PARTIAL]${RESET} data.csv present, images missing"
    else
        echo -e "  - Fashion Dataset       : ${YELLOW}[MISSING]${RESET} Download required"
    fi

    # SQLite Database
    if [[ -f "$DB_FILE" && -f "$PYTHON_BIN" ]]; then
        local prod_cnt=0
        prod_cnt="$("$PYTHON_BIN" -c "import sqlite3; c=sqlite3.connect('$DB_FILE'); print(c.execute('SELECT COUNT(*) FROM products;').fetchone()[0]); c.close()" 2>/dev/null || echo 0)"
        if [[ "$prod_cnt" -gt 0 ]]; then
            echo -e "  - SQLite Database       : ${GREEN}[OK]${RESET} fashion.db ($prod_cnt records)"
        else
            echo -e "  - SQLite Database       : ${YELLOW}[EMPTY]${RESET} fashion.db has 0 records"
        fi
    else
        echo -e "  - SQLite Database       : ${YELLOW}[MISSING]${RESET} fashion.db not generated"
    fi

    # FAISS Index
    if [[ -f "$INDEX_FILE" && -f "$PYTHON_BIN" ]]; then
        local idx_cnt=0
        idx_cnt="$("$PYTHON_BIN" -c "import faiss; idx=faiss.read_index('$INDEX_FILE'); print(idx.ntotal)" 2>/dev/null || echo 0)"
        if [[ "$idx_cnt" -gt 0 ]]; then
            echo -e "  - FAISS Vector Index    : ${GREEN}[OK]${RESET} fashion.index ($idx_cnt vectors)"
        else
            echo -e "  - FAISS Vector Index    : ${YELLOW}[EMPTY]${RESET} fashion.index has 0 vectors"
        fi
    else
        echo -e "  - FAISS Vector Index    : ${YELLOW}[MISSING]${RESET} fashion.index not generated"
    fi

    # Ollama Service
    local o_stat o_ver o_srv
    o_stat="$(get_ollama_status)"
    o_ver="$(echo "$o_stat" | grep -o 'VERSION=[^ ]*' | cut -d= -f2 || echo 'UNKNOWN')"
    o_srv="$(echo "$o_stat" | grep -o 'SERVICE=[^ ]*' | cut -d= -f2 || echo 'UNKNOWN')"
    if [[ "$o_ver" != "NOT_INSTALLED" && "$o_srv" == "active" ]]; then
        echo -e "  - Ollama Service        : ${GREEN}[OK]${RESET} Active (v$o_ver)"
    elif [[ "$o_ver" != "NOT_INSTALLED" ]]; then
        echo -e "  - Ollama Service        : ${YELLOW}[INSTALLED]${RESET} Service state: $o_srv"
    else
        echo -e "  - Ollama Service        : ${YELLOW}[NOT INSTALLED]${RESET}"
    fi

    echo -e "------------------------------------------------------------------------------"
}

# ------------------------------------------------------------------------------
# Core Setup and Execution Routines
# ------------------------------------------------------------------------------
install_dependencies() {
    echo -e "\n${BOLD}${BLUE}==> Installing Direct Python Dependencies (unpinned)${RESET}"
    ensure_venv

    local pm
    pm="$(detect_pkg_manager)"
    echo "[INFO] Dependencies to install: ${DIRECT_DEPENDENCIES[*]}"

    if [[ "$pm" == "uv" ]]; then
        echo -e "${CYAN}[INFO] Using uv package manager...${RESET}"
        uv pip install --python "$PYTHON_BIN" "${DIRECT_DEPENDENCIES[@]}"
    elif [[ -f "$PIP_BIN" ]]; then
        echo -e "${CYAN}[INFO] Using pip in virtual environment...${RESET}"
        "$PIP_BIN" install --upgrade pip
        "$PIP_BIN" install "${DIRECT_DEPENDENCIES[@]}"
    else
        echo -e "${RED}[ERROR] Neither uv nor pip is available in the virtual environment.${RESET}" >&2
        return 1
    fi
    echo -e "${GREEN}[SUCCESS] Dependencies installed.${RESET}"
}

download_dataset_only() {
    echo -e "\n${BOLD}${BLUE}==> Fashion Dataset Download${RESET}"
    ensure_venv
    if ! check_dependencies_installed; then
        echo "[INFO] Installing missing dependencies before download..."
        install_dependencies
    fi

    local force_flag=""
    if [[ "${1:-}" == "--force" ]]; then
        force_flag="--force"
    fi

    "$PYTHON_BIN" "$DATA_DIR/download_data.py" $force_flag
}

build_database_only() {
    echo -e "\n${BOLD}${BLUE}==> SQLite Database and FAISS Index Generation${RESET}"
    ensure_venv
    if ! check_dependencies_installed; then
        echo "[INFO] Installing missing dependencies before build..."
        install_dependencies
    fi

    local force_flag=""
    if [[ "${1:-}" == "--force" ]]; then
        force_flag="--force"
    fi

    "$PYTHON_BIN" "$DATA_DIR/build_database.py" $force_flag
}

setup_ollama_interactive() {
    echo -e "\n${BOLD}${BLUE}==> Ollama Service Configuration${RESET}"
    if [[ ! -f "$OLLAMA_SCRIPT" ]]; then
        echo -e "${RED}[ERROR] Ollama setup script not found at $OLLAMA_SCRIPT${RESET}"
        return 1
    fi

    local status
    status="$(get_ollama_status)"
    echo "[INFO] Current Ollama status: $status"

    echo ""
    echo "Select Ollama action:"
    echo "  1) Install / Start Ollama and pull standard model (llama3)"
    echo "  2) Install / Start Ollama and pull compact model (qwen2.5-coder:3b)"
    echo "  3) Install / Start Ollama daemon only (no models)"
    echo "  4) Pull specified model"
    echo "  5) Return / Skip"
    read -rp "Selection [1-5]: " o_choice

    case "$o_choice" in
        1)
            bash "$OLLAMA_SCRIPT" --models llama3
            configure_env "llama3"
            ;;
        2)
            bash "$OLLAMA_SCRIPT" --models qwen2.5-coder:3b
            configure_env "qwen2.5-coder:3b"
            ;;
        3)
            bash "$OLLAMA_SCRIPT" --no-models
            configure_env "gpt-oss:120b-cloud"
            ;;
        4)
            read -rp "Enter model name (e.g. gemma3:1b): " cust_model
            if [[ -n "$cust_model" ]]; then
                bash "$OLLAMA_SCRIPT" --models "$cust_model"
                configure_env "$cust_model"
            fi
            ;;
        *)
            echo "[INFO] Skipping Ollama configuration."
            ;;
    esac
}

configure_env() {
    local model_name="${1:-gpt-oss:120b-cloud}"
    if [[ ! -f "$ENV_FILE" ]]; then
        echo "[INFO] Initializing .env configuration..."
        cat << EOF > "$ENV_FILE"
# Psyduck Configuration
OLLAMA_HOST=127.0.0.1
OLLAMA_PORT=11434
OLLAMA_MODEL=$model_name
EOF
        echo -e "${GREEN}[SUCCESS] Created .env with OLLAMA_MODEL=$model_name${RESET}"
    else
        echo -e "${CYAN}[INFO] Existing .env detected.${RESET}"
    fi
}

verify_deployment_readiness() {
    echo -e "\n${BOLD}${CYAN}==> Verifying Deployment Readiness${RESET}"
    local ready=true

    # Python binary
    if [[ -f "$PYTHON_BIN" ]]; then
        echo -e "  [PASS] Python virtual environment verified"
    else
        echo -e "  ${RED}[FAIL] Python virtual environment missing${RESET}"
        ready=false
    fi

    # Dependencies
    if check_dependencies_installed; then
        echo -e "  [PASS] Direct dependencies verified"
    else
        echo -e "  ${RED}[FAIL] Missing Python dependencies${RESET}"
        ready=false
    fi

    # Dataset CSV
    if [[ -f "$CSV_FILE" && -s "$CSV_FILE" ]]; then
        echo -e "  [PASS] data.csv found ($(du -h "$CSV_FILE" | cut -f1))"
    else
        echo -e "  ${RED}[FAIL] data/data.csv missing or empty${RESET}"
        ready=false
    fi

    # Image files
    local img_cnt=0
    if [[ -d "$IMAGES_DIR" ]]; then
        img_cnt="$(ls -1 "$IMAGES_DIR" 2>/dev/null | wc -l)"
    fi
    if [[ $img_cnt -gt 0 ]]; then
        echo -e "  [PASS] data/database images verified ($img_cnt files)"
    else
        echo -e "  ${YELLOW}[WARN] data/database has 0 images. Image serving will be disabled.${RESET}"
    fi

    # Database and FAISS Index
    if [[ -f "$DB_FILE" && -f "$INDEX_FILE" ]]; then
        echo -e "  [PASS] fashion.db and fashion.index present"
    else
        echo -e "  ${RED}[FAIL] SQLite database or FAISS index missing${RESET}"
        ready=false
    fi

    # Web assets
    if [[ -f "$SCRIPT_DIR/src/templates/index.html" && -f "$SCRIPT_DIR/src/static/style.css" && -f "$SCRIPT_DIR/src/app.py" ]]; then
        echo -e "  [PASS] Web application assets present (src/app.py, templates, static)"
    else
        echo -e "  ${RED}[FAIL] Web application assets missing in src/${RESET}"
        ready=false
    fi

    # Ollama reachability
    if curl -s --connect-timeout 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        echo -e "  [PASS] Ollama daemon reachable on 127.0.0.1:11434"
    else
        echo -e "  ${YELLOW}[WARN] Ollama daemon not responding on 127.0.0.1:11434${RESET}"
    fi

    echo "------------------------------------------------------------------------------"
    if [[ "$ready" == "true" ]]; then
        echo -e "${GREEN}${BOLD}[SUCCESS] Psyduck server is ready for deployment.${RESET}"
        return 0
    else
        echo -e "${RED}${BOLD}[ERROR] Deployment prerequisites not satisfied.${RESET}"
        return 1
    fi
}

run_full_setup() {
    echo -e "${BOLD}${BLUE}Starting Full Automated Setup...${RESET}"
    install_dependencies
    download_dataset_only
    build_database_only
    if [[ -z "${CI:-}" && -t 0 ]]; then
        setup_ollama_interactive
    else
        echo "[INFO] Automated non-interactive mode: configuring environment..."
        configure_env "gpt-oss:120b-cloud"
    fi
    verify_deployment_readiness
}

launch_server() {
    ensure_venv
    local model_flag=""
    if [[ -n "${1:-}" ]]; then
        model_flag="--model $1"
        echo -e "${CYAN}[INFO] Model override: $1${RESET}"
    fi
    echo -e "${GREEN}${BOLD}[INFO] Launching Psyduck server at http://127.0.0.1:8000...${RESET}"
    exec "$PYTHON_BIN" "$SCRIPT_DIR/src/app.py" --host 0.0.0.0 --port 8000 $model_flag
}

# ------------------------------------------------------------------------------
# Setup Submenu
# ------------------------------------------------------------------------------
setup_submenu() {
    while true; do
        clear 2>/dev/null || true
        print_banner
        echo -e "${BOLD}--- Setup & Configuration Submenu ---${RESET}\n"
        echo -e "  ${CYAN}1)${RESET} Full Automated Setup (Dependencies, Dataset, Database, Ollama)"
        echo -e "  ${CYAN}2)${RESET} Install / Update Python Dependencies (unpinned, uv/pip)"
        echo -e "  ${CYAN}3)${RESET} Download Fashion Dataset only (KaggleHub -> data/)"
        echo -e "  ${CYAN}4)${RESET} Build Database & FAISS Index only (data.csv -> fashion.db & index)"
        echo -e "  ${CYAN}5)${RESET} Setup Ollama (install daemon & pull models)"
        echo -e "  ${CYAN}6)${RESET} Verify Deployment Readiness"
        echo -e "  ${CYAN}7)${RESET} Return to Main Menu"
        echo ""
        read -rp "Enter choice [1-7]: " s_choice

        case "$s_choice" in
            1)
                run_full_setup
                read -rp "Press Enter to continue..."
                ;;
            2)
                install_dependencies
                read -rp "Press Enter to continue..."
                ;;
            3)
                echo "1) Download dataset (skip if exists)"
                echo "2) Force re-download dataset"
                read -rp "Selection [1-2]: " d_opt
                if [[ "$d_opt" == "2" ]]; then
                    download_dataset_only --force
                else
                    download_dataset_only
                fi
                read -rp "Press Enter to continue..."
                ;;
            4)
                echo "1) Build database (skip if exists)"
                echo "2) Force rebuild database and index"
                read -rp "Selection [1-2]: " b_opt
                if [[ "$b_opt" == "2" ]]; then
                    build_database_only --force
                else
                    build_database_only
                fi
                read -rp "Press Enter to continue..."
                ;;
            5)
                setup_ollama_interactive
                read -rp "Press Enter to continue..."
                ;;
            6)
                verify_deployment_readiness || true
                read -rp "Press Enter to continue..."
                ;;
            7|q|Q)
                return 0
                ;;
            *)
                echo -e "${RED}[ERROR] Invalid option: $s_choice${RESET}"
                sleep 1
                ;;
        esac
    done
}

# ------------------------------------------------------------------------------
# Main TUI Menu
# ------------------------------------------------------------------------------
tui_main_menu() {
    while true; do
        clear 2>/dev/null || true
        print_banner
        print_system_status

        echo -e "${BOLD}Main Menu:${RESET}"
        echo -e "  ${CYAN}1)${RESET} Start Psyduck Server (uvicorn app:app)"
        echo -e "  ${CYAN}2)${RESET} Setup & Maintenance Menu (Submenu)"
        echo -e "  ${CYAN}3)${RESET} Verify Deployment Readiness"
        echo -e "  ${CYAN}4)${RESET} Configure Active Model / Environment (.env)"
        echo -e "  ${CYAN}5)${RESET} Exit"
        echo ""
        read -rp "Enter choice [1-5]: " choice

        case "$choice" in
            1)
                read -rp "Enter model override (or press Enter for default): " m_input
                launch_server "$m_input"
                ;;
            2)
                setup_submenu
                ;;
            3)
                verify_deployment_readiness || true
                read -rp "Press Enter to continue..."
                ;;
            4)
                read -rp "Enter target Ollama model name (e.g. llama3, qwen2.5-coder:3b): " m_cfg
                if [[ -n "$m_cfg" ]]; then
                    configure_env "$m_cfg"
                fi
                read -rp "Press Enter to continue..."
                ;;
            5|q|Q)
                echo "[INFO] Exiting server manager."
                exit 0
                ;;
            *)
                echo -e "${RED}[ERROR] Invalid option: $choice${RESET}"
                sleep 1
                ;;
        esac
    done
}

# ------------------------------------------------------------------------------
# CLI Flag Dispatcher
# ------------------------------------------------------------------------------
usage() {
    cat << EOF
Usage: ./run.sh [OPTIONS] [MODEL]

Primary Options:
  --run [model]          Start the Psyduck web application server (optional model argument)
  --setup                Enter the interactive setup submenu or execute setup
  --full                 Run the full automated setup end-to-end non-interactively
  --deps                 Install or update unpinned direct dependencies (uv or pip)
  --download-data        Download and place fashion dataset into data/
  --force-download       Force re-download of dataset
  --build-db             Build SQLite database and FAISS vector index
  --force-build-db       Force rebuild of database and FAISS index
  --ollama [model]       Configure Ollama service and pull model (default: llama3)
  --check                Verify all deployment prerequisites
  -h, --help             Show this help message

Running ./run.sh without arguments in a terminal opens the interactive TUI.
EOF
}

if [[ $# -eq 0 ]]; then
    if [[ -t 0 ]]; then
        tui_main_menu
    else
        # In non-interactive mode without arguments, launch server
        launch_server
    fi
    exit 0
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)
            model_arg="${2:-}"
            if [[ -n "$model_arg" && ! "$model_arg" =~ ^-- ]]; then
                shift
            else
                model_arg=""
            fi
            launch_server "$model_arg"
            exit 0
            ;;
        --setup)
            if [[ -t 0 ]]; then
                setup_submenu
            else
                run_full_setup
            fi
            exit 0
            ;;
        --full)
            run_full_setup
            exit 0
            ;;
        --deps)
            install_dependencies
            exit 0
            ;;
        --download-data)
            download_dataset_only
            exit 0
            ;;
        --force-download)
            download_dataset_only --force
            exit 0
            ;;
        --build-db)
            build_database_only
            exit 0
            ;;
        --force-build-db)
            build_database_only --force
            exit 0
            ;;
        --ollama)
            model_arg="${2:-llama3}"
            if [[ "$#" -gt 1 && ! "$2" =~ ^-- ]]; then
                shift
            fi
            bash "$OLLAMA_SCRIPT" --models "$model_arg"
            configure_env "$model_arg"
            exit 0
            ;;
        --check)
            verify_deployment_readiness
            exit $?
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            # If a model name is passed directly, e.g.: ./run.sh llama3
            if [[ ! "$1" =~ ^- ]]; then
                launch_server "$1"
                exit 0
            else
                echo "[ERROR] Unknown argument: $1" >&2
                usage >&2
                exit 2
            fi
            ;;
    esac
done
