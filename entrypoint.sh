#!/bin/bash
set -e

# Create secrets.toml from environment variables with proper TOML formatting
# Using a temporary file approach to avoid permission issues
TMP_SECRETS=$(mktemp)
cat > $TMP_SECRETS << EOF
# API Keys
TAVILY_API_KEY = "${TAVILY_API_KEY}"
GROQ_API_KEY = "${GROQ_API_KEY}"
OPENAI_API_KEY = "${OPENAI_API_KEY}"
EOF

# Safely move the temp file to the destination
cat $TMP_SECRETS > /app/.streamlit/secrets.toml
rm $TMP_SECRETS

# Start Streamlit
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0