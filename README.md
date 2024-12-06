# Aadhaar Customer Service Chatbot - DigiMitraAI

An intelligent chatbot system for Aadhaar-related queries using RAG (Retrieval Augmented Generation) and LLM technologies.

## Features
- PDF based knowledge base management
- RAG implementation using FAISS vector store
- LLM integration with ChatGPT
- Audio input processing
- Gradio user interface

## Prerequisites
1. Python 3.8+
2. Google Cloud Platform Account
3. OpenAI API Key
4. Git


## Google Cloud Setup
1. Create a Google Cloud Project:
   ```bash
   # Visit https://console.cloud.google.com
   # Create a new project or select existing project
   ```

2. Enable Required APIs:
   - Cloud Speech-to-Text API
   - Cloud Text-to-Speech API
   - Cloud Translation API

3. Create Service Account:
   - Go to IAM & Admin > Service Accounts
   - Create new service account
   - Grant following roles:
     * Cloud Speech-to-Text Admin
     * Cloud Text-to-Speech Admin
     * Cloud Translation API Editor

4. Generate Credentials:
   - Create new key for service account (JSON format)
   - Download JSON file
   - Place in `data/credentials/google-cloud-credentials.json`

## Installation Steps

1. Clone Repository:
```bash
git clone 
cd digimitraai
```

2. Create Virtual Environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Environment Variables:
```bash
# Create .env file
echo "OPENAI_API_KEY=your-openai-key-here" > .env
echo "GOOGLE_APPLICATION_CREDENTIALS=data/credentials/google-cloud-credentials.json" >> .env
```

5. Create Required Directories:
```bash
mkdir -p data/pdf_docs data/vector_store data/credentials
```

6. Add PDF Documents:
```bash
# Copy your PDF documents to data/pdf_docs/
cp path/to/your/pdfs/* data/pdf_docs/
```

7. Initialize Knowledge Base:
```bash
python initialize_pdf_knowledge_base.py
```

## Running the Application

1. Start the Application:
```bash
python frontend/app.py
```

2. Access the Interface:
- Open browser at http://localhost:7860

## Maintenance

1. Update Knowledge Base:
```bash
# Add new PDFs to data/pdf_docs/
python initialize_pdf_knowledge_base.py
```

2. Monitor Google Cloud Usage:
- Visit Google Cloud Console
- Check API usage quotas
- Monitor billing

## Troubleshooting

1. Google Cloud Credentials Issues:
```bash
# Verify credentials file location
ls -l data/credentials/google-cloud-credentials.json

# Check environment variable
echo $GOOGLE_APPLICATION_CREDENTIALS
```

2. Vector Store Issues:
```bash
# Clear vector store and reinitialize
rm -rf data/vector_store/*
python initialize_pdf_knowledge_base.py
```

3. Audio Processing Issues:
```bash
# Check API enablement
gcloud services list

# Verify audio file formats
# Supported formats: WAV, MP3
```

## Required Python Packages
```
langchain>=0.1.0
langchain-openai>=0.0.2
faiss-cpu>=1.7.4
openai>=1.3.0
gradio>=4.4.0
python-dotenv>=1.0.0
google-cloud-speech>=2.6.0
google-cloud-texttospeech>=2.12.0
google-cloud-translate>=2.0.1
pymupdf>=1.22.5
numpy>=1.24.0
pandas>=2.0.0
tiktoken>=0.5.1
```

## System Requirements
- Memory: Minimum 4GB RAM
- Storage: Minimum 1GB free space
- Internet connection for API access

## Additional Notes
1. Keep Google Cloud credentials secure
2. Regularly update PDF documents
3. Monitor API usage and costs
4. Back up vector store regularly
5. Keep track of API quotas and limits

## Support
For issues related to:
- Google Cloud: Check Google Cloud Documentation
- FAISS/Vector Store: Check FAISS Documentation
- Language Models: Check OpenAI Documentation
