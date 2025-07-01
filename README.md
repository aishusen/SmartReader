📘 SmartReader
SmartReader is a local, privacy-focused AI assistant that helps you interact with your documents. Simply upload a PDF or text file, and ask any question about the content. SmartReader will intelligently retrieve relevant parts and answer you — all powered locally by open-source language models!

🚀 Features
📄 Upload PDF/Text files

❓ Ask questions about file content

🧠 Uses LangChain with Hugging Face's FLAN-T5 model

📚 Context-aware retrieval with vector embeddings

🔍 Shows source pages used in the answer

🛡️ Runs locally — no data sent to the cloud

🖥️ Demo
🗨️ “Upload your syllabus PDF and ask: What are the exam topics?”
SmartReader responds with a direct, clear answer — citing page sources too!

<!-- (Optional: Add your own gif or screenshot here) -->

🧰 Built With
Python

LangChain

HuggingFace Transformers

Chainlit

Chroma Vector Store

🛠️ How to Run Locally
Clone the repo:

bash
Copy
Edit
git clone https://github.com/aishusen/SmartReader.git
cd SmartReader
Create and activate virtual environment:

bash
Copy
Edit
python -m venv venv
venv\Scripts\activate   # On Windows
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the app:

bash
Copy
Edit
chainlit run app.py --port 8001
Open your browser:
Visit 👉 http://localhost:8001

📂 File Structure
bash
Copy
Edit
SmartReader/
│
├── app.py               # Main application logic
├── requirements.txt     # Python dependencies
├── README.md            # Project overview
├── .gitignore           # Git ignored files
├── .chainlit/           # Chainlit settings
├── venv/                # Virtual environment (excluded from Git)
🙋‍♀️ Author
Made with 💡 by Aishwarya S

🌐 License
This project is licensed under the MIT License — feel free to use, modify, and share.
