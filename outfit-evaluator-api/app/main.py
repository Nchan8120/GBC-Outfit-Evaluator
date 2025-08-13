"""
FastAPI application with all endpoints for the Outfit Evaluator API
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
import asyncio
from datetime import datetime

# Import our components
from app.config import (
    API_TITLE, API_DESCRIPTION, API_VERSION,
    OCCASIONS, CLASS_NAMES
)
from app.models import OutfitAnalyzer, LLMSuggestionGenerator
from app.services import model_loader
from app.utils import file_handler

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
outfit_analyzer = None
llm_generator = None

# Pydantic models for request/response
class AnalysisRequest(BaseModel):
    """Request model for analysis with user preferences"""
    include_suggestions: bool = True
    user_preferences: Optional[Dict] = None

class SuggestionRequest(BaseModel):
    """Request model for getting suggestions from existing analysis"""
    analysis_result: Dict
    user_preferences: Optional[Dict] = None

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    models: Dict[str, bool]
    device: str
    upload_stats: Dict

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup"""
    global outfit_analyzer, llm_generator
    
    print("🚀 Starting AI Outfit Evaluator API...")
    
    # Load all models
    model_status = model_loader.load_all_models()
    
    # Initialize analyzers
    outfit_analyzer = OutfitAnalyzer()
    llm_generator = LLMSuggestionGenerator()
    
    # Clean up old files
    cleaned_files = file_handler.cleanup_old_files(max_age_hours=24)
    if cleaned_files > 0:
        print(f"🧹 Cleaned up {cleaned_files} old files")
    
    print("✅ API startup complete!")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🛑 Shutting down API...")
    
    # Clean up resources
    if outfit_analyzer:
        outfit_analyzer.cleanup()
    
    # Cleanup temporary files
    file_handler.cleanup_old_files(max_age_hours=0)  # Clean all files
    
    print("✅ Shutdown complete")

# Background task for file cleanup
async def cleanup_file_task(file_path: str):
    """Background task to clean up uploaded file"""
    await asyncio.sleep(1)  # Small delay to ensure processing is complete
    file_handler.cleanup_file(file_path)

# Import for serving HTML
from fastapi.responses import HTMLResponse

# Root endpoint - serve web app
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web application"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Outfit Evaluator</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }

        .main-content {
            padding: 40px;
        }

        .upload-section {
            background: #f8f9ff;
            border: 2px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .upload-section:hover {
            background: #f0f2ff;
            border-color: #5a6fd8;
        }

        .upload-section.dragover {
            background: #e8ecff;
            border-color: #4c63d2;
            transform: scale(1.02);
        }

        .upload-icon {
            font-size: 4em;
            color: #667eea;
            margin-bottom: 20px;
        }

        .file-input {
            display: none;
        }

        .upload-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 1.1em;
            cursor: pointer;
            transition: transform 0.2s;
        }

        .upload-btn:hover {
            transform: translateY(-2px);
        }

        .preview-image {
            max-width: 100%;
            max-height: 300px;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }

        .form-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        .form-group label {
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
        }

        .form-group select,
        .form-group input {
            padding: 12px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s;
        }

        .form-group select:focus,
        .form-group input:focus {
            outline: none;
            border-color: #667eea;
        }

        .analyze-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 18px 40px;
            border-radius: 12px;
            font-size: 1.2em;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            transition: all 0.3s;
            margin-bottom: 30px;
        }

        .analyze-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        }

        .analyze-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .results {
            display: none;
        }

        .score-display {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 30px;
        }

        .score-number {
            font-size: 4em;
            font-weight: bold;
            margin-bottom: 10px;
        }

        .score-text {
            font-size: 1.3em;
            opacity: 0.9;
        }

        .breakdown {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .breakdown-item {
            background: #f8f9ff;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border-left: 4px solid #667eea;
        }

        .breakdown-score {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }

        .suggestions {
            background: #f8f9ff;
            padding: 30px;
            border-radius: 15px;
        }

        .suggestions h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.3em;
        }

        .suggestion-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid #667eea;
        }

        .detected-items {
            background: #f8f9ff;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
        }

        .detected-items h3 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.3em;
        }

        .items-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }

        .item-card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
            transition: transform 0.2s;
        }

        .item-card:hover {
            transform: translateY(-2px);
        }

        .item-name {
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
            text-transform: capitalize;
        }

        .item-confidence {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 15px;
        }

        .colors-section {
            margin-top: 15px;
        }

        .colors-label {
            font-weight: 600;
            color: #555;
            margin-bottom: 8px;
            font-size: 0.9em;
        }

        .color-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .color-tag {
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 500;
            text-transform: capitalize;
        }

        .color-tag.red { background: #e74c3c; }
        .color-tag.blue { background: #3498db; }
        .color-tag.green { background: #27ae60; }
        .color-tag.yellow { background: #f1c40f; color: #333; }
        .color-tag.orange { background: #e67e22; }
        .color-tag.purple { background: #9b59b6; }
        .color-tag.pink { background: #e91e63; }
        .color-tag.brown { background: #8d6e63; }
        .color-tag.black { background: #2c3e50; }
        .color-tag.white { background: #ecf0f1; color: #333; border: 1px solid #bdc3c7; }
        .color-tag.gray, .color-tag.grey { background: #95a5a6; }
        .color-tag.silver { background: #bdc3c7; color: #333; }
        .color-tag.beige { background: #d2b48c; color: #333; }

        .api-info {
            background: #e8f4f8;
            border: 1px solid #bee5eb;
            border-radius: 10px;
            padding: 20px;
            margin-top: 30px;
        }

        .api-info h3 {
            color: #0c5460;
            margin-bottom: 15px;
        }

        .api-endpoints {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }

        .endpoint {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #17a2b8;
        }

        .endpoint-method {
            font-weight: bold;
            color: #17a2b8;
        }

        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 AI Outfit Evaluator</h1>
            <p>Get AI-powered fashion advice and style suggestions</p>
        </div>

        <div class="main-content">
            <div class="upload-section" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">📸</div>
                <h3>Upload Your Outfit Photo</h3>
                <p>Click here or drag and drop an image</p>
                <button type="button" class="upload-btn" style="margin-top: 20px;">Choose File</button>
                <input type="file" id="fileInput" class="file-input" accept="image/*">
                <img id="previewImage" class="preview-image" style="display: none;">
            </div>

            <div class="form-section">
                <div class="form-group">
                    <label for="occasion">Occasion</label>
                    <select id="occasion">
                        <option value="casual_hangout">Casual Hangout</option>
                        <option value="job_interview">Job Interview</option>
                        <option value="date_night">Date Night</option>
                        <option value="work_meeting">Work Meeting</option>
                        <option value="formal_event">Formal Event</option>
                        <option value="beach_vacation">Beach/Vacation</option>
                        <option value="night_out">Night Out</option>
                        <option value="business_casual">Business Casual</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="style_preference">Style Preference (Optional)</label>
                    <select id="style_preference">
                        <option value="">Select style...</option>
                        <option value="minimalist">Minimalist</option>
                        <option value="bold">Bold & Colorful</option>
                        <option value="classic">Classic</option>
                        <option value="trendy">Trendy</option>
                        <option value="edgy">Edgy</option>
                    </select>
                </div>
            </div>

            <button id="analyzeBtn" class="analyze-btn" disabled>
                Analyze My Outfit
            </button>

            <div class="error" id="errorMessage"></div>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <h3>Analyzing your outfit...</h3>
                <p>This may take a few seconds</p>
            </div>

            <div class="results" id="results">
                <div class="score-display">
                    <div class="score-number" id="scoreNumber">0</div>
                    <div class="score-text" id="scoreText">Style Score</div>
                </div>

                <div class="detected-items" id="detectedItems">
                    <h3>👕 Detected Items & Colors</h3>
                    <div class="items-grid" id="itemsGrid">
                        <!-- Detected items will be populated here -->
                    </div>
                </div>

                <div class="breakdown" id="breakdown">
                    <!-- Score breakdown will be populated here -->
                </div>

                <div class="suggestions" id="suggestions">
                    <h3>💡 Style Suggestions</h3>
                    <div id="suggestionsList">
                        <!-- Suggestions will be populated here -->
                    </div>
                </div>
            </div>

            <div class="api-info">
                <h3>🔗 API Endpoints</h3>
                <div class="api-endpoints">
                    <div class="endpoint">
                        <div class="endpoint-method">POST /analyze</div>
                        <div>Analyze outfit with AI suggestions</div>
                    </div>
                    <div class="endpoint">
                        <div class="endpoint-method">GET /occasions</div>
                        <div>Get available occasions</div>
                    </div>
                    <div class="endpoint">
                        <div class="endpoint-method">GET /health</div>
                        <div>Check API health status</div>
                    </div>
                    <div class="endpoint">
                        <div class="endpoint-method">GET /docs</div>
                        <div>Interactive API documentation</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let selectedFile = null;

        // File input handling
        document.getElementById('fileInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                selectedFile = file;
                const reader = new FileReader();
                reader.onload = function(e) {
                    const preview = document.getElementById('previewImage');
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(file);
                document.getElementById('analyzeBtn').disabled = false;
            }
        });

        // Drag and drop
        const uploadSection = document.querySelector('.upload-section');
        
        uploadSection.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadSection.classList.add('dragover');
        });

        uploadSection.addEventListener('dragleave', function(e) {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
        });

        uploadSection.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (file.type.startsWith('image/')) {
                    selectedFile = file;
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        const preview = document.getElementById('previewImage');
                        preview.src = e.target.result;
                        preview.style.display = 'block';
                    };
                    reader.readAsDataURL(file);
                    document.getElementById('analyzeBtn').disabled = false;
                }
            }
        });

        // Analyze button
        document.getElementById('analyzeBtn').addEventListener('click', async function() {
            if (!selectedFile) return;

            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('occasion', document.getElementById('occasion').value);
            formData.append('include_suggestions', 'true');
            
            const stylePreference = document.getElementById('style_preference').value;
            if (stylePreference) {
                formData.append('user_style_preference', stylePreference);
            }

            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            document.getElementById('errorMessage').style.display = 'none';
            document.getElementById('analyzeBtn').disabled = true;

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const result = await response.json();
                displayResults(result);

            } catch (error) {
                console.error('Error:', error);
                document.getElementById('errorMessage').textContent = 'Error analyzing outfit: ' + error.message;
                document.getElementById('errorMessage').style.display = 'block';
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('analyzeBtn').disabled = false;
            }
        });

        function displayResults(result) {
            // Display score
            document.getElementById('scoreNumber').textContent = result.style_score;
            document.getElementById('scoreText').textContent = result.contextual_feedback;

            // Display detected items
            displayDetectedItems(result.detected_items);

            // Display breakdown
            const breakdown = document.getElementById('breakdown');
            breakdown.innerHTML = '';
            
            const breakdownData = [
                { name: 'Contextual', score: result.scoring_breakdown.clip_contextual },
                { name: 'Color Harmony', score: result.scoring_breakdown.color_harmony },
                { name: 'Completeness', score: result.scoring_breakdown.item_completeness },
                { name: 'Coherence', score: result.scoring_breakdown.style_coherence }
            ];

            breakdownData.forEach(item => {
                const div = document.createElement('div');
                div.className = 'breakdown-item';
                div.innerHTML = `
                    <div class="breakdown-score">${item.score}</div>
                    <div>${item.name}</div>
                `;
                breakdown.appendChild(div);
            });

            // Display suggestions
            const suggestionsList = document.getElementById('suggestionsList');
            suggestionsList.innerHTML = '';

            if (result.whats_working) {
                const div = document.createElement('div');
                div.className = 'suggestion-item';
                div.innerHTML = `<strong>✅ What's Working:</strong> ${result.whats_working}`;
                suggestionsList.appendChild(div);
            }

            if (result.areas_for_improvement) {
                const div = document.createElement('div');
                div.className = 'suggestion-item';
                div.innerHTML = `<strong>🔧 Areas for Improvement:</strong> ${result.areas_for_improvement}`;
                suggestionsList.appendChild(div);
            }

            if (result.specific_suggestions && result.specific_suggestions.length > 0) {
                result.specific_suggestions.forEach(suggestion => {
                    const div = document.createElement('div');
                    div.className = 'suggestion-item';
                    div.innerHTML = `<strong>💡 Suggestion:</strong> ${suggestion}`;
                    suggestionsList.appendChild(div);
                });
            }

            if (result.occasion_tips) {
                const div = document.createElement('div');
                div.className = 'suggestion-item';
                div.innerHTML = `<strong>🎯 Occasion Tips:</strong> ${result.occasion_tips}`;
                suggestionsList.appendChild(div);
            }

            document.getElementById('results').style.display = 'block';
        }

        function displayDetectedItems(detectedItems) {
            const itemsGrid = document.getElementById('itemsGrid');
            itemsGrid.innerHTML = '';

            if (!detectedItems || detectedItems.length === 0) {
                itemsGrid.innerHTML = '<p>No clothing items detected</p>';
                return;
            }

            detectedItems.forEach(item => {
                const itemCard = document.createElement('div');
                itemCard.className = 'item-card';

                // Create color tags
                let colorTagsHTML = '';
                if (item.colors && item.colors.length > 0) {
                    colorTagsHTML = item.colors.map(color => 
                        `<span class="color-tag ${color.name.toLowerCase().replace('_', '-')}">${color.name.replace('_', ' ')}</span>`
                    ).join('');
                } else {
                    colorTagsHTML = '<span class="color-tag">No colors detected</span>';
                }

                // Get item emoji
                const itemEmojis = {
                    'shirt': '👔',
                    'pants': '👖', 
                    'jacket': '🧥',
                    'dress': '👗',
                    'skirt': '🩱',
                    'shorts': '🩳',
                    'shoe': '👟',
                    'bag': '👜',
                    'hat': '👒',
                    'sunglass': '🕶️'
                };

                const emoji = itemEmojis[item.class] || '👕';

                itemCard.innerHTML = `
                    <div class="item-name">${emoji} ${item.class}</div>
                    <div class="item-confidence">Confidence: ${Math.round(item.confidence * 100)}%</div>
                    <div class="colors-section">
                        <div class="colors-label">Detected Colors:</div>
                        <div class="color-tags">
                            ${colorTagsHTML}
                        </div>
                    </div>
                `;

                itemsGrid.appendChild(itemCard);
            });
        }
    </script>
</body>
</html>
    """

# API info endpoint
@app.get("/api/info")
async def api_info():
    """API information endpoint"""
    return {
        "message": "AI Outfit Evaluator API",
        "version": API_VERSION,
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "analyze": "POST /analyze - Analyze outfit image",
            "suggest": "POST /suggest - Get LLM suggestions",
            "occasions": "GET /occasions - Get available occasions",
            "health": "GET /health - Health check"
        }
    }

# Get available occasions
@app.get("/occasions")
async def get_occasions():
    """Get list of available occasions for outfit analysis"""
    return {
        "occasions": list(OCCASIONS.keys()),
        "descriptions": OCCASIONS,
        "total_count": len(OCCASIONS)
    }

# Get class names
@app.get("/classes")
async def get_class_names():
    """Get list of clothing classes that can be detected"""
    return {
        "classes": CLASS_NAMES,
        "total_count": len(CLASS_NAMES)
    }

# Main analysis endpoint
@app.post("/analyze")
async def analyze_outfit(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Outfit image file"),
    occasion: str = Form(..., description="Occasion type"),
    include_suggestions: bool = Form(True, description="Include LLM suggestions"),
    user_style_preference: Optional[str] = Form(None, description="User style preference"),
    user_budget: Optional[str] = Form(None, description="Budget level"),
    avoid_items: Optional[str] = Form(None, description="Comma-separated items to avoid")
):
    """
    Analyze an outfit image and optionally get AI suggestions
    
    - **file**: Upload an image file (JPG, PNG, etc.)
    - **occasion**: Choose from available occasions (use /occasions endpoint)
    - **include_suggestions**: Whether to include AI-generated suggestions
    - **user_style_preference**: Optional style preference (e.g., "minimalist", "bold")
    - **user_budget**: Optional budget level (e.g., "low", "moderate", "high")
    - **avoid_items**: Optional comma-separated list of items to avoid
    """
    
    start_time = datetime.now()
    temp_file_path = None
    
    try:
        # Validate occasion
        if occasion not in OCCASIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid occasion '{occasion}'. Available: {list(OCCASIONS.keys())}"
            )
        
        # Validate and read file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_content = await file.read()
        
        # Validate file
        is_valid, error_message = file_handler.validate_file(
            file_content, 
            file.filename, 
            file.content_type
        )
        
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_message)
        
        # Save uploaded file
        success, message, temp_file_path = file_handler.save_upload(
            file_content, 
            file.filename
        )
        
        if not success:
            raise HTTPException(status_code=500, detail=f"File upload failed: {message}")
        
        # Optimize image for processing
        file_handler.optimize_image(temp_file_path, max_width=1024, max_height=1024)
        
        # Analyze outfit
        if not outfit_analyzer:
            raise HTTPException(status_code=503, detail="Outfit analyzer not available")
        
        analysis_result = outfit_analyzer.analyze_outfit(temp_file_path, occasion)
        
        # Add user preferences to result
        user_preferences = {}
        if user_style_preference:
            user_preferences['style_preference'] = user_style_preference
        if user_budget:
            user_preferences['budget'] = user_budget
        if avoid_items:
            user_preferences['avoid_items'] = [item.strip() for item in avoid_items.split(',')]
        
        if user_preferences:
            analysis_result['user_preferences'] = user_preferences
        
        # Generate LLM suggestions if requested
        if include_suggestions and llm_generator:
            try:
                final_result = llm_generator.generate_suggestions(analysis_result, user_preferences)
            except Exception as e:
                print(f"LLM suggestion error: {e}")
                final_result = analysis_result
                final_result['suggestion_error'] = str(e)
                final_result['ai_suggestions_available'] = False
        else:
            final_result = analysis_result
            final_result['ai_suggestions_available'] = False
        
        # Add request metadata
        processing_time = (datetime.now() - start_time).total_seconds()
        final_result.update({
            'request_id': f"{start_time.strftime('%Y%m%d_%H%M%S')}_{hash(file.filename) % 10000}",
            'timestamp': start_time.isoformat(),
            'processing_time_seconds': round(processing_time, 2),
            'file_info': file_handler.get_file_info(temp_file_path)
        })
        
        # Schedule file cleanup
        if temp_file_path:
            background_tasks.add_task(cleanup_file_task, temp_file_path)
        
        return JSONResponse(content=final_result)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        if temp_file_path:
            background_tasks.add_task(cleanup_file_task, temp_file_path)
        raise
        
    except Exception as e:
        # Handle unexpected errors
        if temp_file_path:
            background_tasks.add_task(cleanup_file_task, temp_file_path)
        
        print(f"Unexpected error in analyze_outfit: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

# LLM suggestions endpoint
@app.post("/suggest")
async def get_suggestions(request: SuggestionRequest):
    """
    Generate LLM suggestions for an existing outfit analysis
    
    - **analysis_result**: Previous analysis result from /analyze endpoint
    - **user_preferences**: Optional user preferences for personalized suggestions
    """
    
    try:
        if not llm_generator:
            raise HTTPException(status_code=503, detail="LLM service not available")
        
        # Generate suggestions
        enhanced_result = llm_generator.generate_suggestions(
            request.analysis_result, 
            request.user_preferences
        )
        
        # Add metadata
        enhanced_result['suggestion_timestamp'] = datetime.now().isoformat()
        
        return JSONResponse(content=enhanced_result)
        
    except Exception as e:
        print(f"Error generating suggestions: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Suggestion generation failed: {str(e)}"
        )

# Quick tips endpoint
@app.get("/tips/{occasion}")
async def get_quick_tips(occasion: str):
    """
    Get quick styling tips for a specific occasion
    
    - **occasion**: Occasion type
    """
    
    if occasion not in OCCASIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid occasion '{occasion}'. Available: {list(OCCASIONS.keys())}"
        )
    
    try:
        if llm_generator:
            tips = llm_generator.get_quick_tips(occasion, [])
        else:
            # Fallback tips
            tips = [
                f"Dress appropriately for {OCCASIONS[occasion]}",
                "Ensure good fit and cleanliness",
                "Coordinate colors thoughtfully"
            ]
        
        return {
            "occasion": occasion,
            "occasion_description": OCCASIONS[occasion],
            "tips": tips
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting tips: {str(e)}")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check of the API and all services
    """
    
    try:
        # Check model status
        model_status = model_loader.get_model_status()
        
        # Check analyzer status
        analyzer_ready = outfit_analyzer is not None
        llm_ready = llm_generator is not None
        
        # Get upload statistics
        upload_stats = file_handler.get_upload_stats()
        
        # Overall health status
        critical_services = [
            model_status['yolo_loaded'],
            model_status['clip_loaded'], 
            analyzer_ready
        ]
        
        overall_status = "healthy" if all(critical_services) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            models={
                "yolo": model_status['yolo_loaded'],
                "clip": model_status['clip_loaded'],
                "gemini": model_status['gemini_loaded'],
                "analyzer_ready": analyzer_ready,
                "llm_ready": llm_ready
            },
            device=model_status['device'],
            upload_stats=upload_stats
        )
        
    except Exception as e:
        print(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": f"The endpoint {request.url.path} does not exist",
            "available_endpoints": [
                "/", "/docs", "/occasions", "/classes", 
                "/analyze", "/suggest", "/tips/{occasion}", "/health"
            ]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

# Export the app
__all__ = ["app"]