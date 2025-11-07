# Psyduck

**Psyduck** is a fashion designer and outfit selector for various occasions.

## Features

- Smart outfit suggestions for multiple scenarios
- Easy-to-use interface
- Curated fashion database

## Language Composition

- **HTML** (80%)
- **Python** (10.3%)
- **CSS** (9.7%)

---

## Getting Started (Linux/WSL)

Follow the steps below to set up and run Psyduck on **Linux/WSL**.

### 1. Clone the Repository

```sh
git clone https://github.com/ladybug-me/psyduck.git
cd psyduck
```

### 2. Create & Activate Python Virtual Environment

```sh
python -m venv .psyduck
source .psyduck/bin/activate    # use 'source .psyduck/bin/activate.fish' if you use the fish shell
```

### 3. Install Requirements

```sh
pip install -r requirements.txt
```

### 4. Download Fashion Data

```sh
python3 data/download_data.py
```

### 5. Move Downloaded Data

Replace `<path to downloaded data>` with your actual file path.

```sh
mv <path to downloaded data> data/
```

### 6. Build the Database

```sh
cd data
python3 build_database.py
cd ../
```

### 7. Start the Application

```sh
python3 app.py
```

### 8. Visit the Website

Open your browser and go to:

```
http://127.0.0.1:8000
```

---

## Special Instructions for WSL Users

- **Network settings:**  
  In your WSL options (via Windows Terminal/WSL Integration or in your virtualization manager), set **networking to 'Mirrored'** and **enable Host address loopback**.  
  This ensures that **the app.py running inside WSL can access Ollama running on Windows at port 11434**.

---

## Setting Up Ollama Desktop

Psyduck uses Ollama for local AI model inference.

1. **Download & Install Ollama Desktop**  
   - Download Ollama from [https://ollama.com/download](https://ollama.com/download)  
   - Install it on your OS.
   - Note you need to signin in the ollama app in order to access the gpt-oss:120b-cloud model (can be changed in app.py)

2. **Expose Ollama to Your Network**  
   - Open the Ollama Desktop settings.
   - Ensure **"Expose Ollama to your network"** is enabled.
     - This allows calls from WSL to access Ollama (default port: 11434).
   - Restart Ollama Desktop if needed.

---

## Directory Structure

After setup, your project folder will look like this:

```
psyduck/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── data.csv
│   ├── fashion.db
│   ├── fashion.index
│   ├── download_data.py
│   ├── build_database.py
│   └── database/
│       ├── photo1.jpg
│       ├── photo2.jpg
│       └── ... (all fashion photos here)
├── <other Python or HTML/CSS files>
```

- `data/` contains the fashion data, database, index, and downloaded photos.
- `database/` (inside `data/`) houses all fashion photos.

---

## Notes

- Replace `<path to downloaded data>` with the actual path to your downloaded data files.
- Before running `python3 app.py`, ensure the database is built and all data files are present.
- For support on other platforms, see future updates or raise an issue.

---

Happy styling!
