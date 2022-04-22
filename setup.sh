mkdir -p ~/.streamlit/

echo "
[theme]
base='light'
primaryColor='#0ca48b'
secondaryBackgroundColor='#ecf2f3'
[server]
port = $PORT
enableCORS = false
headless = true
" > ~/.streamlit/config.toml
