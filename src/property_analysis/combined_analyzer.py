"""
Combined Property Analyzer: Text + Vision Integration

Orchestrates multi-modal analysis with intelligent stage selection:
  Stage 1: Text analysis (all properties)
  Stage 2: CLIP vision (all properties, free)
  Stage 3: Claude Vision (top candidates only, budget-conscious)
"""
from typing import List, Optional, Dict
import numpy as np
from pathlib import Path
import mlflow

from .text_analyzer import PropertyTextAnalyzer
from .vision_analyzer import VisionAnalyzer, VisionAnalysis
from .schemas import (
    PropertyAnalysis, DetectedFeature, QueryRequirement, 
    MatchResult, CombinedAnalysis
)
from .vision_decision_agent import VisionDecisionAgent


class CombinedPropertyAnalyzer:
    """
    Multi-modal property analyzer with budget-aware vision analysis
    
    Scoring strategy:
      - Text features: 40%
      - Vision features: 30% (if Stage 3 used)
      - Semantic similarity: 30% (text + CLIP)
    """
    
    def __init__(
        self,
        text_backend: str = "api",
        enable_vision: bool = True,
        vision_mode: str = "claude_primary",  # claude_primary, qwen_primary, claude_only, qwen_only
        vision_budget: int = 300,  # Max Claude Vision calls
        qwen_confidence_threshold: float = 0.7,
        cache_dir: Optional[str] = None
    ):
        # Text analyzer
        self.text_analyzer = PropertyTextAnalyzer(
            backend=text_backend,
            cache_dir=cache_dir
        )
        
        # Vision analyzer
        self.enable_vision = enable_vision
        self.vision_analyzer = None
        self.vision_decision_agent = None
        
        if enable_vision:
            self.vision_analyzer = VisionAnalyzer(
                mode=vision_mode,
                qwen_confidence_threshold=qwen_confidence_threshold,
                cache_dir=f"{cache_dir}/vision" if cache_dir else None,
                max_claude_calls=vision_budget
            )
            self.vision_decision_agent = VisionDecisionAgent()
        
        print(f"✅ CombinedAnalyzer ready (vision: {enable_vision}, mode: {vision_mode})")

    def analyze_with_vision_description(
       self,
       property_id: str,
       description: str,
       image_urls: List[str],
       user_query: str
        ) -> PropertyAnalysis:
       """
       Analiza propiedad usando Claude Vision para generar descripción desde fotos
       
       La descripción de fotos se concatena con descripción de texto,
       luego el text_analyzer detecta features de ambas fuentes.
       """
       # Generate photo description
       photo_description = self.vision_analyzer.generate_photo_description(
           image_urls=image_urls,
           max_images=3,
           user_query=user_query
       )
       
       # Combine descriptions
       combined_description = f"{description}\n\n{photo_description}"
       
       # Analyze combined text (features from both sources)
       analysis = self.text_analyzer.analyze(
           property_id=property_id,
           description=combined_description,
           generate_embedding=True
       )
       
       return analysis

    def analyze_property_stage1(
        self,
        property_id: str,
        description: str,
        image_urls: List[str],
        generate_embeddings: bool = True
    ) -> tuple[PropertyAnalysis, Optional[VisionAnalysis]]:
        """
        Stage 1: Text + CLIP analysis (free, always runs)
        
        Returns: (text_analysis, vision_analysis)
        """
        # Text analysis
        text_analysis = self.text_analyzer.analyze(
            property_id=property_id,
            description=description,
            generate_embedding=generate_embeddings
        )
        
        # Vision analysis (CLIP only)
        vision_analysis = None
        if self.enable_vision and image_urls:
            vision_analysis = self.vision_analyzer.analyze_property_stage1(
                property_id=property_id,
                image_urls=image_urls,
                max_images=3
            )
        
        return text_analysis, vision_analysis
    
    def analyze_property_stage2(
        self,
        property_id: str,
        image_urls: List[str],
        query_text: str,
        target_features: List[str]
    ) -> VisionAnalysis:
        """
        Stage 2: Claude Vision for detailed feature extraction
        COSTS MONEY - use only on filtered properties!
        
        Args:
            target_features: Features to look for (e.g., ["entrada_independiente"])
        """
        if not self.enable_vision:
            raise ValueError("Vision analysis not enabled")
        
        return self.vision_analyzer.analyze_property_stage2(
            property_id=property_id,
            image_urls=image_urls,
            query_text=query_text,
            target_features=target_features,
            max_images=2  # Control cost
        )
    
    def compute_combined_score(
        self,
        text_analysis: PropertyAnalysis,
        vision_analysis: Optional[VisionAnalysis],
        query_text: str,
        requirements: List[QueryRequirement],
        use_vision_features: bool = False
    ) -> MatchResult:
        """
        Compute combined score from text and vision
        
        Args:
            use_vision_features: If True, use Claude Vision features (Stage 2)
                                If False, use CLIP similarity only (Stage 1)
        """
        # 1. Text feature matching
        text_match = self.text_analyzer.match_against_query(
            property_analysis=text_analysis,
            query_text=query_text,
            required_features=requirements,
            feature_weight=1.0,  # We'll reweight below
            semantic_weight=0.0  # Computed separately
        )
        
        text_feature_score = text_match.feature_match_score
        text_semantic_score = text_match.semantic_similarity_score
        matched_features = text_match.matched_features.copy()
        missing = text_match.missing_requirements.copy()
        
        # 2. Vision analysis
        vision_feature_score = 0.0
        vision_semantic_score = 0.0
        
        if vision_analysis:
            # CLIP semantic similarity (always available)
            if vision_analysis.get_avg_embedding() is not None:
                vision_semantic_score = self.vision_analyzer.compute_image_text_similarity(
                    vision_analysis.get_avg_embedding(),
                    query_text
                )
            
            # Claude Vision features (if Stage 2 was run)
            if use_vision_features and vision_analysis.detected_features:
                vision_feature_score = self._score_vision_features(
                    vision_analysis.detected_features,
                    requirements
                )
                
                # Merge features with text
                matched_features = self._merge_features(
                    matched_features,
                    vision_analysis.detected_features
                )
                
                # Update missing requirements
                vision_feature_names = {f.name for f in vision_analysis.detected_features}
                missing = [m for m in missing if m not in vision_feature_names]
        
        # 3. Combined scoring
        if use_vision_features:
            # Stage 2: Full vision features available
            # 40% text features, 30% vision features, 15% text semantic, 15% vision semantic
            final_score = (
                text_feature_score * 0.40 +
                vision_feature_score * 0.30 +
                text_semantic_score * 0.15 +
                vision_semantic_score * 0.15
            )
        else:
            # Stage 1: Only CLIP similarity
            # 50% text features, 30% text semantic, 20% vision semantic
            final_score = (
                text_feature_score * 0.50 +
                text_semantic_score * 0.30 +
                vision_semantic_score * 0.20
            )
        
        return MatchResult(
            property_id=text_analysis.property_id,
            feature_match_score=(text_feature_score + vision_feature_score) / 2,
            semantic_similarity_score=(text_semantic_score + vision_semantic_score) / 2,
            final_score=final_score,
            matched_features=matched_features,
            missing_requirements=missing
        )
    
    def _score_vision_features(
        self,
        vision_features: List[DetectedFeature],
        requirements: List[QueryRequirement]
    ) -> float:
        """Score vision features against requirements"""
        if not requirements:
            return 0.0
        
        score = 0.0
        total_importance = sum(r.importance for r in requirements)
        
        for req in requirements:
            # Check if vision detected this feature
            for vf in vision_features:
                if self._features_match(req.feature_name, vf.name):
                    score += vf.confidence * req.importance
                    break
        
        return score / total_importance if total_importance > 0 else 0.0
    
    def _features_match(self, req_name: str, detected_name: str) -> bool:
        """Check if feature names match (fuzzy)"""
        req_words = set(req_name.lower().replace('_', ' ').split())
        det_words = set(detected_name.lower().replace('_', ' ').split())
        
        # Exact match
        if req_name.lower() == detected_name.lower():
            return True
        
        # Partial overlap (>50%)
        overlap = len(req_words & det_words)
        return overlap > 0 and overlap / max(len(req_words), len(det_words)) > 0.5
    
    def _merge_features(
        self,
        text_features: List[DetectedFeature],
        vision_features: List[DetectedFeature]
    ) -> List[DetectedFeature]:
        """
        Merge text and vision features intelligently
        
        Rules:
        - If same feature in both, take higher confidence
        - Otherwise, keep all features
        """
        merged = {}
        
        # Add text features
        for f in text_features:
            merged[f.name] = f
        
        # Add or update with vision features
        for f in vision_features:
            if f.name in merged:
                # Same feature - take higher confidence
                if f.confidence > merged[f.name].confidence:
                    merged[f.name] = f
            else:
                # New feature from vision
                merged[f.name] = f
        
        return list(merged.values())
    
    def analyze_batch_stage1(
        self,
        properties: List[dict],
        query_text: str,
        requirements: List[QueryRequirement],
        use_vision_agent: bool = True,
        log_to_mlflow: bool = True
    ) -> List[Dict]:
        """
        Batch analysis with intelligent vision usage
       
       Args:
           use_vision_agent: Si True, usa LLM para decidir cuándo usar vision

        Returns: List of scored properties with rankings
        """

        # Validate inputs
        if not properties:
            print("⚠️  No properties to analyze")
            return []
        
        if not query_text:
            print("⚠️  Empty query text")
            return []

        results = []

        # First pass: Text-only analysis for all properties
        print("Stage 1: Text-only analysis...")
        
        for i, prop in enumerate(properties, 1):
            prop_id = prop.get('id', f'prop_{i}')
            print(f"  {i}/{len(properties)}: {prop_id}")
            
            try:
                # Text analysis
                print(f"    Analyzing text for {prop_id}...")
                text_analysis = self.text_analyzer.analyze(
                    property_id=prop_id,
                    description=prop.get('description', '')
                )
                print(f"    Text analysis OK")
                
                # Text match
                print(f"    Matching against query...")
                text_match = self.text_analyzer.match_against_query(
                    property_analysis=text_analysis,
                    query_text=query_text,
                    required_features=requirements
                )
                print(f"    Match OK, score: {text_match.final_score:.3f}")
                
                results.append({
                    'property': prop,
                    'text_analysis': text_analysis,
                    'text_match': text_match,
                    'score': text_match.final_score,
                    'needs_vision': False  # Updated later
                })
                print(f"    Added to results")
            except Exception as e:
               import traceback
               print(f"  ⚠️  Error analyzing {prop_id}: {e}")
               print(f"  Traceback:")
               traceback.print_exc()
               continue
        
        if not results:
            print("⚠️  No successful analyses")
            return []
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)

        # Decide which properties need vision
        if use_vision_agent and self.enable_vision:
            print("\nStage 2: Vision decision agent...")
           
            try:
                # Debug info
                print(f"  Number of results: {len(results)}")
                print(f"  Top 5 scores: {[r['score'] for r in results[:5]]}")
                properties_with_scores = [
                    {'property_id': r['property']['id'], 'text_score': r['score']}
                    for r in results
                ]
            
                decision = self.vision_decision_agent.should_use_vision(
                    query=query_text,
                    text_score=None,  # Will check scores internally
                    top_candidates_scores=[r['score'] for r in results[:min(5, len(results))]]
                )
            
                print(f"  Decision: {'USE VISION' if decision['use_vision'] else 'SKIP VISION'}")
                print(f"  Confidence: {decision['confidence']:.2f}")
                print(f"  Reasoning: {decision['reasoning']}")
                
                if decision['use_vision']:
                    # Analyze with vision (top candidates only)
                    top_n = min(20, len(results))
                    print(f"\nStage 3: Vision analysis for top {top_n} candidates...")
                    
                    for i, result in enumerate(results[:top_n], 1):
                        prop = result['property']
                        prop_id = prop.get('id')
                        
                        print(f"  {i}/{top_n}: {prop_id}")
                        
                        # Generate photo description + reanalyze
                        print(f"    Analyzing {len(prop.get('images', []))} images...")
                        # Versión actualizada con user_query
                        photo_description = self.vision_analyzer.generate_photo_description(
                            image_urls=prop.get('images', []),
                            max_images=3,
                            user_query=query_text  # Pasar el query completo
                        )
                        
                        # Combine descriptions
                        combined_description = f"{prop.get('description', '')}\n\n{photo_description}"
                        
                        # Analyze combined text
                        enhanced_analysis = self.text_analyzer.analyze(
                            property_id=prop_id,
                            description=combined_description,
                            generate_embedding=True
                        )
                        
                        # Re-score with enhanced features
                        enhanced_match = self.text_analyzer.match_against_query(
                            property_analysis=enhanced_analysis,
                            query_text=query_text,
                            required_features=requirements
                        )
                        
                        # Update result
                        result['text_analysis'] = enhanced_analysis
                        result['text_match'] = enhanced_match
                        result['score'] = enhanced_match.final_score
                        result['needs_vision'] = True
                        result['photo_description'] = photo_description
                    
                    # Re-sort after vision analysis
                    results.sort(key=lambda x: x['score'], reverse=True)
                
            except Exception as e:
                print(f"⚠️  Vision agent error: {e}")
                print("  Continuing with text-only results...")
            
        # MLflow logging
        if log_to_mlflow and mlflow.active_run():
            mlflow.log_metric("total_properties", len(results))
            if results:
                mlflow.log_metric("top_score", results[0]['score'])
            mlflow.log_metric("avg_score", np.mean([r['score'] for r in results]))
            mlflow.log_metric("vision_used", sum(1 for r in results if r['needs_vision']))

        
        return results
    
    def enhance_top_candidates_stage2(
        self,
        candidates: List[Dict],
        query_text: str,
        requirements: List[QueryRequirement],
        top_n: int = 50,
        log_to_mlflow: bool = True
    ) -> List[Dict]:
        """
        Enhance top candidates with Claude Vision (Stage 2)
        COSTS MONEY - selective use!
        
        Args:
            candidates: Results from Stage 1
            top_n: Number of top candidates to enhance
        """
        if not self.enable_vision:
            print("⚠️  Vision not enabled, skipping Stage 2")
            return candidates
        
        # Check budget
        budget_status = self.vision_analyzer.get_budget_status()
        if budget_status['remaining'] < top_n:
            print(f"⚠️  Insufficient budget: {budget_status['remaining']} calls remaining")
            top_n = budget_status['remaining']
        
        print(f"\n🔍 Stage 2: Analyzing top {top_n} with Claude Vision")
        
        # Extract target features from requirements
        target_features = [r.feature_name for r in requirements if r.importance > 0.5]
        
        enhanced = []
        for i, candidate in enumerate(candidates[:top_n], 1):
            prop = candidate['property']
            prop_id = prop.get('id', f'prop_{i}')
            
            print(f"  {i}/{top_n}: {prop_id}")
            
            # Claude Vision analysis
            vision_analysis_detailed = self.analyze_property_stage2(
                property_id=prop_id,
                image_urls=prop.get('images', []),
                query_text=query_text,
                target_features=target_features
            )
            
            # Re-score with vision features
            match_enhanced = self.compute_combined_score(
                text_analysis=candidate['text_analysis'],
                vision_analysis=vision_analysis_detailed,
                query_text=query_text,
                requirements=requirements,
                use_vision_features=True  # Stage 2: Use features
            )
            
            enhanced.append({
                'property': prop,
                'text_analysis': candidate['text_analysis'],
                'vision_analysis': vision_analysis_detailed,
                'match': match_enhanced,
                'score': match_enhanced.final_score,
                'stage1_score': candidate['score']
            })
        
        # Keep remaining candidates from Stage 1
        enhanced.extend(candidates[top_n:])
        
        # Re-sort by new scores
        enhanced.sort(key=lambda x: x['score'], reverse=True)
        
        # MLflow logging
        if log_to_mlflow and mlflow.active_run():
            mlflow.log_metric("stage2_num_enhanced", min(top_n, len(candidates)))
            mlflow.log_metric("stage2_top_score", enhanced[0]['score'])
            
            budget_status = self.vision_analyzer.get_budget_status()
            mlflow.log_metric("stage2_total_cost_eur", budget_status['total_cost_eur'])
            mlflow.log_metric("stage2_remaining_budget", budget_status['remaining'])
        
        return enhanced
    
    def get_status(self) -> Dict:
        """Get analyzer status"""
        status = {
            'text_backend': self.text_analyzer.backend,
            'vision_enabled': self.enable_vision
        }
        
        if self.enable_vision:
            status['vision_budget'] = self.vision_analyzer.get_budget_status()
        
        return status