"""
Application entry point for the AI Outfit Evaluator API
"""

import uvicorn
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.config import API_HOST, API_PORT

def main():
    """Main entry point"""
    print("🚀 Starting AI Outfit Evaluator API Server...")
    print(f"📁 Project root: {project_root}")
    print(f"🌐 Server will be available at: http://{API_HOST}:{API_PORT}")
    print(f"📚 API Documentation: http://{API_HOST}:{API_PORT}/docs")
    print("=" * 60)
    
    # Check if required directories exist
    required_dirs = ['Models', 'uploads']
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            print(f"📁 Creating directory: {dir_path}")
            dir_path.mkdir(exist_ok=True)
    
    # Check if model file exists
    model_path = project_root / "Models" / "best.pt"
    if not model_path.exists():
        print("⚠️  WARNING: Model file 'Models/best.pt' not found!")
        print("   Please ensure your trained YOLO model is placed at Models/best.pt")
        print("   The API will start but outfit detection will not work.")
        print()
    
    # Set environment variables if not set
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  WARNING: GEMINI_API_KEY environment variable not set!")
        print("   LLM suggestions will use fallback mode.")
        print("   Set your Gemini API key to enable AI suggestions.")
        print()
    
    try:
        # Run the FastAPI app
        uvicorn.run(
            "app.main:app",
            host=API_HOST,
            port=API_PORT,
            reload=True,  # Auto-reload on code changes (disable in production)
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()