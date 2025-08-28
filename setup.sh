#!/bin/bash

# Check if venv exists in home directory
if [ ! -d "$HOME/venv" ]; then
    echo "❌ Error: Virtual environment not found in home directory"
    echo "Please create a virtual environment in your home directory using:"
    echo "python -m venv ~/venv"
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Error: Virtual environment is not activated"
    echo "Please activate your virtual environment first using:"
    echo "source ~/venv/bin/activate"
    exit 1
fi

# Check if Gunicorn is installed
if ! pip show gunicorn >/dev/null 2>&1; then
    echo "Installing Gunicorn..."
    pip install gunicorn
    echo "✅ Gunicorn installed successfully"
else
    echo "✅ Gunicorn is already installed"
fi

# Check for .env file
if [ -f ".env" ]; then
    echo "✅ Found .env file in root directory"
else
    echo "❌ Error: .env file not found in root directory"
    echo "Please create a .env file with the following variables:"
    echo "OPENAI_ENV_KEY=your_openai_key"
    echo "EQX_ON_ERROR=nan"
    echo "MAIL_USERNAME=your_email"
    echo "MAIL_PASSWORD=your_app_password"
    echo "FLASK_SECRET_KEY=your_secret_key"
    exit 1
fi

echo -e "\n📋 Server Setup Instructions"
echo "============================="

echo -e "\n🔄 Gunicorn Setup"
echo "----------------"
echo "1. Copy service file:"
echo "   sudo cp server_config_files/diffeqparamfitting.service /etc/systemd/system/"
echo "2. Reload systemd:"
echo "   sudo systemctl daemon-reload"
echo "3. Enable and start service:"
echo "   sudo systemctl enable diffeqparamfitting"
echo "   sudo systemctl start diffeqparamfitting"
echo "4. Check status:"
echo "   sudo systemctl status diffeqparamfitting"

echo -e "\n🌐 Apache Setup"
echo "-------------"
echo "1. Deactivate virtual environment:"
echo "   deactivate"
echo "2. Install and start Apache:"
echo "   sudo apt-get install apache2"
echo "   sudo service apache2 start"
echo "3. Configure Apache:"
echo "   cd /etc/apache2/sites-available"
echo "   sudo rm default-ssn.conf 000-default.conf"
echo "   sudo cp server_config_files/diffeqparamfitting.com.conf . "
echo "4. Enable required modules and site:"
echo "   sudo a2enmod ssl headers proxy proxy_http"
echo "   sudo a2dissite 000-default.conf"
echo "   sudo a2ensite diffeqparamfitting.com.conf"
echo "5. Restart Apache and check status:"
echo "   sudo systemctl restart apache2"
echo "   sudo systemctl status apache2"


echo -e "\n🌐 SSL Setup"
echo "1. Install certbot: sudo snap install --classic certbot"
echo "2. sudo ln -s /snap/bin/certbot /usr/bin/certbot"
echo "3. run: sudo certbot --apache"

echo "NOTE: CHANGE .env TO have SERVER_NAME=yourdomain.com!!!! in production"

echo "NOTE:apache2ctl -S SHOULD ONLY SHOW 1 VIRTUALHOST!! from one file only, ensure this"

echo -e "\n✅ Setup completed successfully"
