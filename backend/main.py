"""
FastAPI Backend for Real Estate AI Finder

Endpoints:
- POST /search: Search properties with natural language
- GET /health: Health check
- GET /status: System status
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
import sys
from pathlib import Path
import os
import traceback
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.query_parser.parser import QueryParser
from src.data.scraper import FotocasaScraper
from src.property_analysis.combined_analyzer import CombinedPropertyAnalyzer
from src.property_analysis.schemas import QueryRequirement

# Initialize FastAPI
app = FastAPI(
    title="Real Estate AI Finder API",
    description="Intelligent property search with multi-modal AI analysis",
    version="1.0.0"
)

# CORS middleware (allow Streamlit frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class SearchRequest(BaseModel):
    """Search request from user"""
    query: str = Field(..., description="Natural language search query in Spanish")
    api_key: str = Field(..., description="Anthropic API key")
    backend: Literal["api", "local"] = Field(default="api", description="Text analysis backend")
    vision_mode: Optional[Literal["claude_only", "qwen_only", "claude_primary", "qwen_primary"]] = Field(
        default="claude_primary",
        description="Vision analysis mode (None for text-only)"
    )
    use_vision_agent: bool = Field(default=True, description="Use LLM agent to decide when to analyze images")
    max_results: int = Field(default=10, description="Max properties to return", ge=1, le=50)
    skip_scrape: bool = Field(default=False, description="Use cached data instead of scraping")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "Local con entrada independiente Barcelona máximo 250mil",
                "api_key": "sk-ant-api03-...",
                "backend": "api",
                "vision_mode": "qwen_only",
                "use_vision_agent": True,
                "max_results": 10,
                "skip_scrape": False
            }
        }


class PropertyResult(BaseModel):
    """Single property result"""
    id: str
    score: float
    price: Optional[int]
    size_m2: Optional[int]
    rooms: Optional[int]
    location: str
    description: str
    images: List[str]
    matched_features: List[str]
    missing_features: List[str]
    had_vision_analysis: bool
    url: Optional[str] = None


class SearchResponse(BaseModel):
    """Search response"""
    success: bool
    properties: List[PropertyResult]
    total_found: int
    query_parsed: dict
    cost_summary: dict
    processing_time_seconds: float
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_api_key(api_key: str) -> bool:
    """Validate Anthropic API key by making a test call"""
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        
        # Simple test call
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10,
            messages=[{"role": "user", "content": "test"}]
        )
        return True
    except Exception as e:
        print(f"API key validation failed: {e}")
        return False


def get_cached_properties(max_properties: int = 50) -> List[dict]:
    """Load cached properties from data/raw"""
    from pathlib import Path
    import json
    
    data_dir = Path("data/raw")
    if not data_dir.exists():
        return []
    
    json_files = sorted(data_dir.glob("fotocasa_*.json"), reverse=True)
    
    properties = []
    for json_file in json_files[:3]:  # Last 3 files
        try:
            with open(json_file) as f:
                data = json.load(f)
                properties.extend(data.get('properties', []))
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    return properties[:max_properties]


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Real Estate AI Finder API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0"
    )


@app.post("/search", response_model=SearchResponse)
async def search_properties(request: SearchRequest):
    """
    Search properties with natural language query
    
    This endpoint:
    1. Parses the query to extract filters
    2. Scrapes properties (or uses cache)
    3. Analyzes properties (text + optional vision)
    4. Ranks and returns top matches
    """
    start_time = datetime.utcnow()
    
    try:
        # Validate API key first
        print(f"Validating API key...")
        if not validate_api_key(request.api_key):
            raise HTTPException(
                status_code=401,
                detail="Invalid Anthropic API key. Please check your key and try again."
            )
        
        print(f"✅ API key valid")
        
        # Set API key in environment for this request
        os.environ["ANTHROPIC_API_KEY"] = request.api_key
        
        # ============================================================
        # STEP 1: Parse Query
        # ============================================================
        print(f"\n[STEP 1] Parsing query: {request.query}")
        query_parser = QueryParser()
        parsed = query_parser.parse(request.query)
        
        print(f"  ✅ Parsed - Direct filters: {parsed.direct_filters.model_dump(exclude_none=True)}")
        print(f"  ✅ Indirect filters: {parsed.indirect_filters.model_dump(exclude_none=True)}")
        
        # ============================================================
        # STEP 2: Get Properties (scrape or cache)
        # ============================================================
        print(f"\n[STEP 2] Getting properties...")
        
        if request.skip_scrape:
            print(f"  📂 Loading from cache...")
            properties = get_cached_properties(max_properties=50)
            print(f"  ✅ Loaded {len(properties)} properties from cache")
        else:
            print(f"  🔍 Scraping Fotocasa...")
            scraper = FotocasaScraper()
            
            # Convert parsed filters to scraper params
            scraper_params = {
                "location": parsed.direct_filters.location or "barcelona-capital",
                "property_type": parsed.direct_filters.property_type or "vivienda",
                "operation": parsed.direct_filters.operation or "comprar",
                "max_pages": 2
            }
            
            if parsed.direct_filters.min_price:
                scraper_params["min_price"] = parsed.direct_filters.min_price
            if parsed.direct_filters.max_price:
                scraper_params["max_price"] = parsed.direct_filters.max_price
            if parsed.direct_filters.min_size_m2:
                scraper_params["min_size_m2"] = parsed.direct_filters.min_size_m2
            if parsed.direct_filters.min_rooms:
                scraper_params["min_rooms"] = parsed.direct_filters.min_rooms
            
            result = scraper.scrape_properties(**scraper_params)
            properties = result.properties
            
            # Save results
            if properties:
                filepath = result.save("data/raw")
                print(f"  ✅ Scraped {len(properties)} properties")
                print(f"  💾 Saved to: {filepath.name}")
        
        if not properties:
            raise HTTPException(
                status_code=404,
                detail="No properties found matching your criteria. Try different filters."
            )
        
        # Limit properties for analysis
        properties_to_analyze = properties[:min(20, len(properties))]
        print(f"  📊 Analyzing {len(properties_to_analyze)} properties")
        
        # ============================================================
        # STEP 3: Analyze Properties
        # ============================================================
        print(f"\n[STEP 3] Analyzing properties (backend: {request.backend}, vision: {request.vision_mode})")
        
        # Initialize analyzer
        analyzer = CombinedPropertyAnalyzer(
            text_backend=request.backend,
            vision_mode=request.vision_mode,
            use_vision=request.vision_mode is not None,
            use_vision_agent=request.use_vision_agent,
            max_claude_calls=100
        )
        
        # Build requirements from indirect filters
        requirements = []
        if parsed.indirect_filters.entrance_type:
            requirements.append(QueryRequirement(
                feature_name="entrada_independiente",
                importance=1.0,
                required=True
            ))
        if parsed.indirect_filters.natural_light:
            requirements.append(QueryRequirement(
                feature_name="luz_natural",
                importance=0.8
            ))
        if parsed.indirect_filters.layout:
            requirements.append(QueryRequirement(
                feature_name=parsed.indirect_filters.layout.replace(' ', '_'),
                importance=0.7
            ))
        
        # Convert Property objects to dicts
        props_as_dicts = []
        for prop in properties_to_analyze:
            if hasattr(prop, 'to_dict'):
                props_as_dicts.append(prop.to_dict())
            else:
                props_as_dicts.append(prop)
        
        # Run analysis
        results = analyzer.analyze_batch_stage1(
            properties=props_as_dicts,
            query_text=request.query,
            query_requirements=requirements if requirements else None,
            top_n=5  # Top 5 get vision analysis
        )
        
        print(f"  ✅ Analysis complete")
        
        # ============================================================
        # STEP 4: Format Response
        # ============================================================
        print(f"\n[STEP 4] Formatting response...")
        
        # Convert to response format
        property_results = []
        for result in results[:request.max_results]:
            prop = result['property']
            match = result['text_match']
            
            property_results.append(PropertyResult(
                id=str(prop['id']),
                score=round(match.final_score, 3),
                price=prop.get('price'),
                size_m2=prop.get('size_m2'),
                rooms=prop.get('rooms'),
                location=prop.get('location', 'Unknown'),
                description=prop.get('description', '')[:200] + "...",  # Truncate
                images=prop.get('images', [])[:3],  # First 3 images
                matched_features=[f.name for f in match.matched_features],
                missing_features=match.missing_requirements,
                had_vision_analysis=result.get('needs_vision', False),
                url=prop.get('url')
            ))
        
        # Get cost summary
        cost_summary = analyzer.get_cost_summary()
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        print(f"  ✅ Returning {len(property_results)} properties")
        print(f"  💰 Total cost: €{cost_summary.get('total_cost_eur', 0):.3f}")
        print(f"  ⏱️  Processing time: {processing_time:.1f}s")
        
        return SearchResponse(
            success=True,
            properties=property_results,
            total_found=len(results),
            query_parsed={
                "direct_filters": parsed.direct_filters.model_dump(exclude_none=True),
                "indirect_filters": parsed.indirect_filters.model_dump(exclude_none=True),
                "confidence": parsed.confidence
            },
            cost_summary=cost_summary,
            processing_time_seconds=round(processing_time, 2),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Startup event"""
    print("=" * 60)
    print("🚀 Real Estate AI Finder API Starting...")
    print("=" * 60)
    print(f"📍 Docs: http://localhost:8000/docs")
    print(f"🏥 Health: http://localhost:8000/health")
    print("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)