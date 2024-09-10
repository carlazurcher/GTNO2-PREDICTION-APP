# Streamlit App

This repository contains a Streamlit app for GTNO2 prediciion. The app leverages machine learning models hosted on Google Drive.

## Prerequisites

- Python 3.8 or higher
- `pip` for installing Python packages

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository
   
   2. **Create and Activate a Virtual Environment**
   - For Windows:
   	python -m venv venv
	.\venv\Scripts\activate
	
	- For Mac/Linux:
	python3 -m venv venv
    source venv/bin/activate
    
    
    3. **Instal dependencies**
    Install the required packages using requirements.txt:
    pip install -r requirements.txt
    
    4. **Run the app**
The app requires machine learning models hosted on Google Drive. The app will automatically download the models at runtime. Start the Streamlit app using the following pip when located in the folder where app is found:
streamlit run app.py

5. **Access the App**
Once the app is running, it will be available at http://localhost:8501 in your web browser.
