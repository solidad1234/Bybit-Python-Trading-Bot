# 🚀 Bybit Python Trading Bot

This repository contains two automated crypto trading bots for **Bybit**, built with Python — one for **spot trading** and another for **futures trading**. Each bot is designed for real-time execution and can be deployed on a VPS for continuous operation.


## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/solidad1234/Bybit-Python-Trading-Bot.git
cd bybit-trading-bot
```

### 2. Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Make sure you have the TA-Lib binary installed. You can use a `.deb` file or compile from source.
```bash
sudo apt update
sudo apt install -y python3-dev build-essential

sudo dpkg -i ~/Downloads/ta-lib_0.6.4_amd64.deb
```

# Install Python dependencies
```bash
pip install ta-lib
```

## ▶️ Running the Bots

### Run Spot Bot
```bash

python3 spot.py
```


### Run Futures Bot
```bash
python3 futures.py
```

## 🧪 Running Tests

### Test Spot Bot
```bash
python3 test.py
```
### Test Futures Bot

```bash
python3 test_futures.py

```

## 🖥️ Deploying on a VPS

### Option 1: Using `screen` 

```bash
sudo apt install screen
screen -S trading-bot
cd ~/bybit-trading-bot
source venv/bin/activate
python3 spot.py  # or python3 futures.py
```
To detach from the screen session:

```bash
Ctrl + A, then D
```

To reattach later:

```bash
screen -r trading-bot
```

### Option 2: Auto-Start on Reboot

Edit crontab:

```bash
crontab -e
```

Add the following line:

```bash
@reboot screen -dmS trading-bot bash -c 'cd ~/bybit-trading-bot && source venv/bin/activate && python3 spot.py >> bot.log 2>&1'
```

*Replace `spot.py` with `futures.py`.*

---

## 🤝 Contributions

Pull requests are welcome. Please ensure any changes are well-tested and clearly documented.

---

## 📬 Contact

For questions or collaborations, feel free to reach out:

**Solidad Kimeu**
📧 [solidadkimeu@gmail.com](mailto:solidadkimeu@gmail.com)

---


