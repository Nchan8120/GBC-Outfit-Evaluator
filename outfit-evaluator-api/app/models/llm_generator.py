"""
LLM suggestion generator using Google Gemini for outfit improvement advice
"""

from typing import Dict, List, Optional
import time

from app.services.model_loader import model_loader

class LLMSuggestionGenerator:
    """Generates fashion suggestions using LLM"""
    
    def __init__(self):
        """Initialize LLM suggestion generator"""
        self.model_available = False
        print("LLMSuggestionGenerator initialized")
    
    def _ensure_model_loaded(self) -> bool:
        """Ensure Gemini model is available"""
        if not self.model_available:
            _, _, _, gemini_model = model_loader.get_models()
            self.model_available = gemini_model is not None
        
        return self.model_available
    
    def generate_suggestions(self, analysis_result: Dict, user_preferences: Optional[Dict] = None) -> Dict:
        """
        Generate outfit suggestions based on analysis results
        
        Args:
            analysis_result: Results from outfit analysis
            user_preferences: Optional user preferences dict
            
        Returns:
            Enhanced analysis result with LLM suggestions
        """
        print("Generating LLM suggestions...")
        start_time = time.time()
        
        if not self._ensure_model_loaded():
            print("Gemini model not available, using fallback suggestions")
            return self._create_fallback_suggestions(analysis_result)
        
        try:
            # Create prompt for LLM
            prompt = self._create_prompt(analysis_result, user_preferences)
            
            # Get Gemini model
            _, _, _, gemini_model = model_loader.get_models()
            
            # Generate response
            response = gemini_model.generate_content(prompt)
            
            if response.text:
                # Parse and structure the response
                enhanced_result = self._parse_response(response.text, analysis_result)
                
                generation_time = time.time() - start_time
                enhanced_result['suggestion_generation_time'] = round(generation_time, 2)
                enhanced_result['ai_suggestions_available'] = True
                
                print(f"LLM suggestions generated in {generation_time:.2f}s")
                return enhanced_result
            else:
                print("Empty response from Gemini, using fallback")
                return self._create_fallback_suggestions(analysis_result)
                
        except Exception as e:
            print(f"Error generating LLM suggestions: {e}")
            return self._create_fallback_suggestions(analysis_result)
    
    def _create_prompt(self, analysis_result: Dict, user_preferences: Optional[Dict]) -> str:
        """Create detailed prompt for Gemini"""
        
        # Extract key information
        score = analysis_result.get('style_score', 0)
        occasion = analysis_result.get('occasion_description', 'casual setting')
        items = analysis_result.get('detected_items', [])
        scoring_breakdown = analysis_result.get('scoring_breakdown', {})
        feedback = analysis_result.get('contextual_feedback', '')
        
        # Build item descriptions with colors
        item_descriptions = []
        for item in items:
            colors = [c['name'] for c in item.get('colors', [])]
            color_str = ', '.join(colors) if colors else 'neutral'
            confidence = item.get('confidence', 0)
            
            item_descriptions.append(
                f"{item['class']} in {color_str} colors (confidence: {confidence:.2f})"
            )
        
        items_text = '\n  - '.join([''] + item_descriptions) if item_descriptions else 'No items detected'
        
        # Add user preferences if provided
        preferences_text = ""
        if user_preferences:
            prefs = []
            if user_preferences.get('style_preference'):
                prefs.append(f"Style preference: {user_preferences['style_preference']}")
            if user_preferences.get('budget'):
                prefs.append(f"Budget: {user_preferences['budget']}")
            if user_preferences.get('avoid_items'):
                avoid_list = ', '.join(user_preferences['avoid_items'])
                prefs.append(f"Items to avoid: {avoid_list}")
            if user_preferences.get('favorite_colors'):
                color_list = ', '.join(user_preferences['favorite_colors'])
                prefs.append(f"Favorite colors: {color_list}")
            
            if prefs:
                preferences_text = f"\n\nUSER PREFERENCES:\n" + '\n'.join(prefs)
        
        # Create comprehensive prompt
        prompt = f"""You are a professional fashion stylist with expertise in creating stylish, contextually appropriate outfits. Analyze this outfit and provide specific, actionable fashion advice.

OUTFIT ANALYSIS:
- Occasion: {occasion}
- Overall Style Score: {score}/10
- Current Assessment: {feedback}

DETECTED CLOTHING ITEMS:{items_text}

DETAILED SCORING BREAKDOWN:
- Contextual Appropriateness: {scoring_breakdown.get('clip_contextual', 0)}/10
- Color Harmony: {scoring_breakdown.get('color_harmony', 0)}/10
- Item Completeness: {scoring_breakdown.get('item_completeness', 0)}/10
- Style Coherence: {scoring_breakdown.get('style_coherence', 0)}/10{preferences_text}

PROVIDE FASHION ADVICE IN THIS EXACT FORMAT:

**WHAT'S WORKING:**
[Identify 1-2 positive aspects of this outfit - be specific about colors, item combinations, or appropriateness for the occasion]

**AREAS FOR IMPROVEMENT:**
[If score < 8, point out 2-3 specific issues. If score >= 8, mention minor tweaks or styling alternatives]

**SPECIFIC SUGGESTIONS:**
[Give 3-4 actionable recommendations such as:]
- Add specific accessories (name exact items like "black leather belt" or "silver watch")
- Change specific clothing pieces ("swap sneakers for dress shoes")
- Adjust colors or patterns ("add a pop of color with a scarf")
- Consider different styling ("tuck in the shirt" or "roll up sleeves")
- Layer appropriately ("add a blazer" or "remove the jacket")

**OCCASION-SPECIFIC TIPS:**
[2-3 tips specifically tailored for {occasion}, considering dress codes and appropriateness]

**SHOPPING SUGGESTIONS:**
[If needed, suggest 1-2 versatile pieces that would improve this and future outfits]

Keep all suggestions practical, specific, and achievable. Focus on improvements that would have the biggest impact on the overall look while respecting the occasion's requirements."""
        
        return prompt
    
    def _parse_response(self, response_text: str, analysis_result: Dict) -> Dict:
        """Parse LLM response into structured format"""
        
        # Start with the original analysis
        enhanced_result = analysis_result.copy()
        
        # Initialize suggestion fields
        suggestions = {
            'whats_working': '',
            'areas_for_improvement': '',
            'specific_suggestions': [],
            'occasion_tips': '',
            'shopping_suggestions': '',
            'raw_llm_response': response_text
        }
        
        # Parse sections from response
        lines = response_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Identify section headers
            if '**WHAT\'S WORKING:**' in line.upper():
                current_section = 'whats_working'
                continue
            elif '**AREAS FOR IMPROVEMENT:**' in line.upper():
                current_section = 'areas_for_improvement'
                continue
            elif '**SPECIFIC SUGGESTIONS:**' in line.upper():
                current_section = 'specific_suggestions'
                continue
            elif '**OCCASION-SPECIFIC TIPS:**' in line.upper():
                current_section = 'occasion_tips'
                continue
            elif '**SHOPPING SUGGESTIONS:**' in line.upper():
                current_section = 'shopping_suggestions'
                continue
            
            # Process content based on current section
            if line and current_section and not line.startswith('**'):
                if current_section == 'specific_suggestions':
                    # Handle list items
                    if line.startswith('- ') or line.startswith('• ') or line.startswith('* '):
                        suggestions[current_section].append(line[2:].strip())
                    elif line and not line.startswith('['):
                        suggestions[current_section].append(line)
                else:
                    # Handle text sections
                    if suggestions[current_section]:
                        suggestions[current_section] += ' ' + line
                    else:
                        suggestions[current_section] = line
        
        # Clean up text sections
        for key in ['whats_working', 'areas_for_improvement', 'occasion_tips', 'shopping_suggestions']:
            suggestions[key] = suggestions[key].strip()
        
        # Add suggestions to result
        enhanced_result.update(suggestions)
        
        return enhanced_result
    
    def _create_fallback_suggestions(self, analysis_result: Dict) -> Dict:
        """Create basic suggestions when LLM is unavailable"""
        
        enhanced_result = analysis_result.copy()
        score = analysis_result.get('style_score', 0)
        occasion = analysis_result.get('occasion', 'casual')
        items = [item['class'] for item in analysis_result.get('detected_items', [])]
        
        # Generate basic suggestions based on score and items
        if score >= 8:
            whats_working = "Your outfit shows excellent coordination and is well-suited for the occasion."
            improvements = "This is already a strong look. Minor adjustments could add extra polish."
            suggestions = [
                "Consider adding a statement accessory to personalize the look",
                "Experiment with different shoe styles for variety",
                "Try layering pieces for added visual interest"
            ]
        elif score >= 6:
            whats_working = "The basic outfit structure works well for this occasion."
            improvements = "Some elements could be refined for better overall impact."
            suggestions = [
                "Focus on improving color coordination between pieces",
                "Consider adding complementary accessories",
                "Pay attention to fit and proportions of garments",
                "Ensure all pieces match the formality level required"
            ]
        else:
            whats_working = "There are elements that provide a good foundation to build upon."
            improvements = "Several aspects could be adjusted for better appropriateness and style."
            suggestions = [
                "Reconsider the color palette for better harmony",
                "Add more occasion-appropriate pieces",
                "Focus on creating better coordination between items",
                "Consider the formality requirements of the occasion"
            ]
        
        # Occasion-specific tips
        if occasion in ['job_interview', 'work_meeting', 'business_casual']:
            occasion_tips = "For professional settings, prioritize conservative colors, proper fit, and polished accessories."
        elif occasion in ['date_night', 'night_out']:
            occasion_tips = "For social occasions, you can be more expressive with colors and accessories while maintaining good taste."
        elif occasion in ['beach_vacation', 'casual_hangout']:
            occasion_tips = "For casual settings, comfort and appropriateness for activities are key, with room for personal expression."
        else:
            occasion_tips = f"For {occasion}, focus on appropriate formality levels and practical considerations."
        
        enhanced_result.update({
            'whats_working': whats_working,
            'areas_for_improvement': improvements,
            'specific_suggestions': suggestions,
            'occasion_tips': occasion_tips,
            'shopping_suggestions': "Consider investing in versatile pieces that can work across multiple occasions.",
            'ai_suggestions_available': False,
            'fallback_used': True
        })
        
        return enhanced_result
    
    def get_quick_tips(self, occasion: str, detected_items: List[str]) -> List[str]:
        """Get quick styling tips based on occasion and items"""
        
        tips = []
        
        if occasion in ['job_interview', 'work_meeting']:
            tips.extend([
                "Ensure all pieces are wrinkle-free and well-fitted",
                "Stick to conservative color palette",
                "Keep accessories minimal and professional"
            ])
            
        elif occasion == 'date_night':
            tips.extend([
                "Add one statement piece to create visual interest",
                "Consider the venue when choosing formality level",
                "Don't forget grooming details - they matter"
            ])
            
        elif occasion == 'beach_vacation':
            tips.extend([
                "Choose breathable fabrics for comfort",
                "Don't forget sun protection accessories",
                "Opt for easy-to-clean materials"
            ])
        
        # Item-specific tips
        if 'jacket' in detected_items:
            tips.append("Ensure jacket fits properly at shoulders and sleeves")
        if 'shoe' in detected_items:
            tips.append("Make sure shoes are clean and appropriate for walking")
        if 'bag' in detected_items:
            tips.append("Choose bag size appropriate for the occasion needs")
        
        return tips[:5]  # Return max 5 tips