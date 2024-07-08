# Healthy Breathe Detection Transformer

We developed a transformer-based AI model that analyzes lung sound recordings to detect respiratory conditions early. By capturing long-range dependencies in spectrograms, our model identifies subtle abnormalities that traditional stethoscope methods often miss.

## Demo Video

Watch the [demo video](https://github.com/tauseef-2611/Healthy-Breath-Detection/blob/main/Healthybreathedemo.mp4) to see Healthy-Breath-Detection in action!


## Installation Instructions

### Frontend (Next.js)

1. **Clone the repository**
   ```sh
   git clone https://github.com/tauseef-2611/Healthy-Breath-Detection.git
   cd Healthy-Breath-Detection/frontend
   ```

2. **Install dependencies**
   ```sh
   npm install
   ```

3. **Run the development server**
   ```sh
   npm run dev
   ```
   Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

### Backend (Flask API)

1. **Navigate to the backend directory**
   ```sh
   cd ../backend
   ```

2. **Create a virtual environment**
   ```sh
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

4. **Run the Flask API**
   ```sh
   flask run
   ```
   The API will be running on [http://localhost:5000](http://localhost:5000).

## Project Structure

```
Healthy-Breath-Detection/
├── frontend/           # Next.js frontend application
│   ├── components/     # React components
│   ├── pages/          # Next.js pages
│   ├── public/         # Public assets
│   └── styles/         # CSS styles
│
├── backend/            # Flask API for model connection
│   ├── app.py          # Main Flask application
│
├── README.md           # Project readMe file
└── requirements.txt    # Backend dependencies
```

## Contributing

We welcome contributions! Please fork the repository and submit pull requests.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
```
