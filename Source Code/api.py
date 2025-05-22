from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import shutil
from tumor_detector import TumorDetector
import json
from datetime import datetime
import sqlite3
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="Brain Tumor Detection API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize tumor detector
try:
    detector = TumorDetector()
    logger.info("TumorDetector initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize TumorDetector: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# Database setup
def init_db():
    try:
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS predictions
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      image_path TEXT,
                      has_tumor INTEGER,
                      confidence REAL,
                      timestamp TEXT)''')
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        logger.error(traceback.format_exc())
        raise

init_db()

@app.post("/predict")
async def predict_tumor(file: UploadFile = File(...)):
    """
    Endpoint to predict tumor in an uploaded image
    """
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Create uploads directory if it doesn't exist
        uploads_dir = "uploads"
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
            logger.info(f"Created uploads directory: {uploads_dir}")
        
        # Save uploaded file
        file_path = os.path.join(uploads_dir, file.filename)
        logger.info(f"Saving file to: {file_path}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info("File saved successfully")
        
        # Make prediction
        logger.info("Making prediction...")
        result = detector.predict(file_path)
        
        if result.get("error"):
            logger.error(f"Prediction error: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])
        
        logger.info(f"Prediction successful: {result}")
        
        # Store prediction in database
        try:
            conn = sqlite3.connect('predictions.db')
            c = conn.cursor()
            c.execute('''INSERT INTO predictions (image_path, has_tumor, confidence, timestamp)
                         VALUES (?, ?, ?, ?)''',
                     (file_path, int(result["has_tumor"]), result["confidence"], 
                      datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
            conn.close()
            logger.info("Prediction stored in database")
        except Exception as e:
            logger.error(f"Failed to store prediction in database: {str(e)}")
            logger.error(traceback.format_exc())
            # Continue even if database storage fails
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/predictions")
async def get_predictions():
    """
    Endpoint to get all predictions
    """
    try:
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        c.execute('SELECT * FROM predictions ORDER BY timestamp DESC')
        predictions = c.fetchall()
        conn.close()
        
        # Convert to list of dictionaries
        result = []
        for pred in predictions:
            result.append({
                "id": pred[0],
                "image_path": pred[1],
                "has_tumor": bool(pred[2]),
                "confidence": pred[3],
                "timestamp": pred[4]
            })
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error getting predictions: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/{prediction_id}")
async def get_prediction(prediction_id: int):
    """
    Endpoint to get a specific prediction
    """
    try:
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        c.execute('SELECT * FROM predictions WHERE id = ?', (prediction_id,))
        prediction = c.fetchone()
        conn.close()
        
        if prediction is None:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        return JSONResponse(content={
            "id": prediction[0],
            "image_path": prediction[1],
            "has_tumor": bool(prediction[2]),
            "confidence": prediction[3],
            "timestamp": prediction[4]
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prediction {prediction_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 