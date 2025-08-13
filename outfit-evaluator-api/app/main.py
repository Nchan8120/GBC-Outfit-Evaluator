"""
FastAPI application with all endpoints for the Outfit Evaluator API
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, List
import asyncio
from datetime import datetime
from pathlib import Path

# Import our components
from .config import (
    API_TITLE, API_DESCRIPTION, API_VERSION,
    OCCASIONS, CLASS_NAMES
)
from .models.outfit_analyzer import OutfitAnalyzer
from .models.llm_generator import LLMSuggestionGenerator
from .services import model_loader
from .utils import file_handler

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

# Mount static files (for CSS, JS, images)
# Note: Path is relative to where the app runs from (project root)
app.mount("/static", StaticFiles(directory="static"), name="static")

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

# Root endpoint - serve web app
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web application"""
    try:
        # Read the HTML file (path relative to project root)
        html_file = Path("templates/index.html")
        if html_file.exists():
            return html_file.read_text(encoding="utf-8")
        else:
            return """
            <html>
                <body>
                    <h1>AI Outfit Evaluator API</h1>
                    <p>HTML template not found. Please check templates/index.html</p>
                    <p><a href="/docs">View API Documentation</a></p>
                </body>
            </html>
            """
    except Exception as e:
        return f"""
        <html>
            <body>
                <h1>AI Outfit Evaluator API</h1>
                <p>Error loading template: {str(e)}</p>
                <p><a href="/docs">View API Documentation</a></p>
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