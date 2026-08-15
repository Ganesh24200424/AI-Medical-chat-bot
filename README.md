# KARE Healthcare Management System

A comprehensive healthcare management system built with Flask, featuring doctor-patient management, appointment scheduling, and video consultations.

## Features

- User authentication and authorization
- Doctor and patient management
- Appointment scheduling
- Video consultations
- Medical report management
- Prescription management
- Admin dashboard

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ahms.git
cd ahms
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize the database:
```bash
flask db upgrade
```

5. Run the application:
```bash
python app.py
```

## Deployment

This application is configured for deployment on Render. Follow these steps:

1. Push your code to a GitHub repository
2. Create a new Web Service on Render
3. Connect your GitHub repository
4. Add the required environment variables
5. Deploy the application

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Font Awesome for icons
- Flask framework
- SQLAlchemy for database management
- WebRTC for video consultations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This software is provided "as is" without warranty of any kind. The authors are not responsible for any damages or liabilities arising from the use of this software.

## Contact

For any queries or support, please contact:
- Email: support@karehealthcare.com
- Website: https://karehealthcare.com 