"""
FastAPI Backend for Real Estate AI Finder

RAILWAY OPTIMIZED: API-only backends, no local GPU dependencies
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
from src.property_analysis.schemas import QueryRequirement, MatchResult

# Initialize FastAPI
app = FastAPI(
    title="Real Estate AI Finder API",
    description="Intelligent property search with AI (Railway deployment)",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class SearchRequest(BaseModel):
    """Search request"""
    query: str = Field(..., description="Natural language search query in Spanish")
    api_key: str = Field(..., description="Anthropic API key")
    vision_mode: Optional[Literal["claude_only", "claude_primary"]] = Field(
        default="claude_primary",
        description="Vision mode (Railway: only claude modes supported)"
    )
    use_vision_agent: bool = Field(default=True, description="Use LLM to decide when to analyze images")
    max_results: int = Field(default=10, description="Max properties to return", ge=1, le=50)
    skip_scrape: bool = Field(default=False, description="Use cached data")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "Local entrada independiente Barcelona máx 250k",
                "api_key": "sk-ant-api03-...",
                "vision_mode": "claude_primary",
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
    photo_description: Optional[str] = None
    match_explanation: Optional[str] = None
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
    mode: str = "railway-api-only"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_api_key(api_key: str) -> bool:
    """Validate Anthropic API key"""
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10,
            messages=[{"role": "user", "content": "test"}]
        )
        return True
    except Exception as e:
        print(f"API key validation failed: {e}")
        return False


def validate_vision_mode(mode: Optional[str]) -> Optional[str]:
    """Validate and correct vision mode for Railway"""
    if mode is None:
        return None
    
    # Railway only supports Claude modes (no GPU for Qwen)
    valid_modes = ["claude_only", "claude_primary"]
    
    if mode in valid_modes:
        return mode
    
    # Auto-correct qwen modes to claude
    if mode in ["qwen_only", "qwen_primary"]:
        print(f"⚠️  Vision mode '{mode}' not supported on Railway (no GPU)")
        print(f"⚠️  Auto-correcting to 'claude_primary'")
        return "claude_primary"
    
    # Unknown mode
    raise ValueError(f"Unknown vision mode: {mode}. Use 'claude_only' or 'claude_primary'")


def get_cached_properties(max_properties: int = 50) -> List[dict]:
    """Load cached properties from data/raw"""
    from pathlib import Path
    import json
    
    data_dir = Path("data/raw")
    if not data_dir.exists():
        return []
    
    json_files = sorted(data_dir.glob("fotocasa_*.json"), reverse=True)
    
    properties = []
    for json_file in json_files[:3]:
        try:
            with open(json_file) as f:
                data = json.load(f)
                properties.extend(data.get('properties', []))
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    return properties[:max_properties]

def _generate_match_explanation(prop: dict, match: MatchResult) -> str:
    """Generate human-readable match explanation"""
    score = match.final_score
    
    if score >= 0.7:
        verdict = "Excelente match"
    elif score >= 0.5:
        verdict = "Buen match"
    elif score >= 0.3:
        verdict = "Match aceptable"
    else:
        verdict = "Match débil"
    
    matched = ", ".join([f.name.replace('_', ' ') for f in match.matched_features[:3]])
    missing = ", ".join(match.missing_requirements[:3])
    
    return f"{verdict} ({score:.2f}). Encontrado: {matched or 'ninguno'}. Falta: {missing or 'ninguno'}."

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Real Estate AI Finder API (Railway)",
        "version": "1.0.0",
        "mode": "api-only",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        mode="railway-api-only"
    )


@app.post("/search", response_model=SearchResponse)
async def search_properties(request: SearchRequest):
    """
    Search properties with natural language
    
    Railway Optimized:
    - Text analysis: Claude API only
    - Vision analysis: Claude Vision API only
    - No local GPU dependencies
    """
    start_time = datetime.utcnow()
    
    try:
        # Validate API key
        print(f"Validating API key...")
        if not validate_api_key(request.api_key):
            raise HTTPException(
                status_code=401,
                detail="Invalid Anthropic API key"
            )
        
        print(f"✅ API key valid")
        
        # Set API key in environment
        os.environ["ANTHROPIC_API_KEY"] = request.api_key
        
        # Validate vision mode for Railway
        vision_mode = validate_vision_mode(request.vision_mode)
        
        # ============================================================
        # STEP 1: Parse Query
        # ============================================================
        print(f"\n[STEP 1] Parsing query: {request.query}")
        query_parser = QueryParser()
        parsed = query_parser.parse(request.query)
        
        print(f"  ✅ Direct filters: {parsed.direct_filters.model_dump(exclude_none=True)}")
        print(f"  ✅ Indirect filters: {parsed.indirect_filters.model_dump(exclude_none=True)}")
        
        # ============================================================
        # STEP 2: Get Properties
        # ============================================================
        print(f"\n[STEP 2] Getting properties...")
        
        if request.skip_scrape:
            print(f"  📂 Loading from cache...")
            properties = get_cached_properties(max_properties=50)
            print(f"  ✅ Loaded {len(properties)} properties")
        else:
            print(f"  🔍 Scraping Fotocasa...")
            scraper = FotocasaScraper()
            
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
            
            if properties:
                filepath = result.save("data/raw")
                print(f"  ✅ Scraped {len(properties)} properties")
                print(f"  💾 Saved to: {filepath.name}")
        
        if not properties:
            raise HTTPException(
                status_code=404,
                detail="No properties found. Try different filters."
            )
        
        limit_analysis = request.max_results + 2
        properties_to_analyze = properties[:min(limit_analysis, len(properties))]
        
        print(f"  📊 Analyzing {len(properties_to_analyze)} properties (Requested top {request.max_results})")
        
        # ============================================================
        # STEP 3: Analyze Properties (API-only)
        # ============================================================
        print(f"\n[STEP 3] Analyzing properties (API-only mode)")
        print(f"  Text: Claude API")
        print(f"  Vision: {vision_mode or 'disabled'}")
        
        # Initialize analyzer (API-only)
        analyzer = CombinedPropertyAnalyzer(
            text_backend="api",  # Force API for Railway
            enable_vision=vision_mode is not None,
            vision_mode=vision_mode or "claude_primary",
            vision_budget=100
        )
        
        # Build requirements
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
        
        # Convert to dicts
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
            requirements=requirements if requirements else None,
            use_vision_agent=request.use_vision_agent,
            log_to_mlflow=False
        )
        
        print(f"  ✅ Analysis complete")
        
        # ============================================================
        # STEP 4: Format Response
        # ============================================================
        print(f"\n[STEP 4] Formatting response...")
        
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
                description=prop.get('description', '')[:200] + "...",
                images=prop.get('images', [])[:3],
                matched_features=[f.name for f in match.matched_features],
                missing_features=match.missing_requirements,
                had_vision_analysis=result.get('needs_vision', False),
                photo_description=result.get('photo_description'),
                match_explanation=self._generate_match_explanation(prop, match),
                url=prop.get('url')
            ))
        
        # Get cost summary
        status = analyzer.get_status()
        cost_summary = {
            'text_backend': 'api',
            'text_cost_per_property': 0.003,
            'total_properties': len(properties_to_analyze)
        }
        
        if 'vision_budget' in status:
            vision_budget = status['vision_budget']
            cost_summary.update({
                'vision_mode': vision_mode,
                'vision_calls_made': vision_budget.get('claude_calls_made', 0),
                'vision_cost_eur': vision_budget.get('total_cost_eur', 0.0),
                'total_cost_eur': (0.003 * len(properties_to_analyze)) + vision_budget.get('total_cost_eur', 0.0)
            })
        else:
            cost_summary['total_cost_eur'] = 0.003 * len(properties_to_analyze)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        print(f"  ✅ Returning {len(property_results)} properties")
        print(f"  💰 Total cost: €{cost_summary.get('total_cost_eur', 0):.3f}")
        print(f"  ⏱️  Time: {processing_time:.1f}s")
        
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
    print("🚀 Real Estate AI Finder API Starting (Railway)")
    print("=" * 60)
    print(f"📍 Mode: API-only (Claude)")
    print(f"📍 Docs: /docs")
    print(f"🏥 Health: /health")
    print("=" * 60)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)