import os
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

# Fetch API Key
API_KEY = os.getenv("API_KEY")
