#  Spending Quest - Class Analyzer

## Course Name
Programming for Financa

## Instructor Name
Dr. Usama Janjua

## 🎮 App Overview
Welcome to **Spending Quest - Class Analyzer**, afinancial classification app built with Streamlit. Users upload their spending data (e.g., groceries, dining, shopping, etc.), and the app analyzes their financial habits using **KMeans clustering** and **data normalization** techniques.

The app sorts users into playful spending classes like the **🛡️ Frugal Knight** or **💎 Lavish Sorcerer**, based on their behavior. It features:

- UI with game-style tooltips
- Radar and bar charts for insights
- CSV upload support and demo mode
- Cluster analysis with optimal number selection
- Downloadable “Character Sheet” report

##  Deployment Link
[Click here to use the app](https://your-username-your-repo-name.streamlit.app)

##  How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/spending-quest-analyzer.git
cd spending-quest-analyzer
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the app
```bash
streamlit run app.py
```
##  Input Format
Upload a .csv file with numeric columns named:

- Groceries
- Dining
- Shopping
- Utilities
- Entertainment
- Savings

Missing columns or non-numeric data will raise errors.




