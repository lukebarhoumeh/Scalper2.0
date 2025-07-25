#!/usr/bin/env python3
"""
Sentiment Analysis System - FinBERT Integration
==============================================
Real-time news and social media sentiment analysis
"""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from collections import deque
import json

# For transformer models
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available - using OpenAI fallback")

from config_unified import OPENAI_API_KEY
from openai import OpenAI

logger = logging.getLogger(__name__)

@dataclass
class SentimentScore:
    """Individual sentiment score"""
    text: str
    score: float  # -1 (bearish) to +1 (bullish)
    confidence: float
    source: str  # 'news', 'twitter', 'reddit', etc.
    timestamp: datetime
    symbols: List[str]  # Mentioned symbols


class SentimentAnalyzer:
    """
    Advanced sentiment analysis using FinBERT and OpenAI
    """
    
    def __init__(self, use_finbert: bool = True):
        self.use_finbert = use_finbert and TRANSFORMERS_AVAILABLE
        
        if self.use_finbert:
            self._init_finbert()
        else:
            self._init_openai()
            
        # Sentiment history
        self.sentiment_history = deque(maxlen=1000)
        self.aggregated_scores = {}
        
        # News sources (in production, use real APIs)
        self.news_sources = {
            'cryptopanic': 'https://cryptopanic.com/api/v1/posts/',
            'newsapi': 'https://newsapi.org/v2/everything',
            'reddit': 'https://www.reddit.com/r/cryptocurrency/new.json'
        }
        
    def _init_finbert(self):
        """Initialize FinBERT model"""
        logger.info("Loading FinBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        self.model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
        self.model.eval()
        
        # Label mapping
        self.label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
    def _init_openai(self):
        """Initialize OpenAI client as fallback"""
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
    def analyze_text(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment of a single text
        Returns: (score, confidence) where score is -1 to +1
        """
        if self.use_finbert:
            return self._analyze_finbert(text)
        else:
            return self._analyze_openai(text)
            
    def _analyze_finbert(self, text: str) -> Tuple[float, float]:
        """Analyze using FinBERT"""
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        # Convert to score
        # negative=0, neutral=1, positive=2
        score = probs[2] - probs[0]  # Positive - Negative
        confidence = max(probs)
        
        return float(score), float(confidence)
        
    def _analyze_openai(self, text: str) -> Tuple[float, float]:
        """Analyze using OpenAI as fallback"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial sentiment analyzer. Analyze the sentiment of crypto market text. Return only a JSON with 'sentiment' (-1 to 1) and 'confidence' (0 to 1)."
                    },
                    {
                        "role": "user",
                        "content": f"Analyze sentiment: {text[:500]}"
                    }
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            result = json.loads(response.choices[0].message.content)
            return result['sentiment'], result['confidence']
            
        except Exception as e:
            logger.error(f"OpenAI sentiment analysis error: {e}")
            return 0.0, 0.0
            
    def extract_symbols(self, text: str, known_symbols: List[str]) -> List[str]:
        """Extract mentioned cryptocurrency symbols"""
        mentioned = []
        
        for symbol in known_symbols:
            # Check for symbol mentions (case insensitive)
            if symbol.upper() in text.upper():
                mentioned.append(symbol)
                
        return mentioned
        
    async def fetch_news(self, symbols: List[str], hours: int = 1) -> List[Dict]:
        """
        Fetch recent news for given symbols
        (In production, implement real API calls)
        """
        # Placeholder - in production, use real news APIs
        # For now, return mock data
        mock_news = [
            {
                'title': f'{symbols[0]} Shows Strong Momentum Amid Market Recovery',
                'description': 'Technical indicators suggest bullish continuation...',
                'source': 'CryptoNews',
                'publishedAt': datetime.now()
            },
            {
                'title': 'Federal Reserve Comments Impact Crypto Markets',
                'description': 'Dovish stance from Fed officials boosts risk assets...',
                'source': 'Bloomberg',
                'publishedAt': datetime.now() - timedelta(minutes=30)
            }
        ]
        
        return mock_news
        
    async def analyze_news_batch(self, news_items: List[Dict], 
                               symbols: List[str]) -> List[SentimentScore]:
        """Analyze a batch of news items"""
        scores = []
        
        for item in news_items:
            # Combine title and description
            text = f"{item.get('title', '')} {item.get('description', '')}"
            
            if text.strip():
                score, confidence = self.analyze_text(text)
                mentioned_symbols = self.extract_symbols(text, symbols)
                
                sentiment_score = SentimentScore(
                    text=text[:200],
                    score=score,
                    confidence=confidence,
                    source=item.get('source', 'unknown'),
                    timestamp=item.get('publishedAt', datetime.now()),
                    symbols=mentioned_symbols
                )
                
                scores.append(sentiment_score)
                self.sentiment_history.append(sentiment_score)
                
        return scores
        
    def aggregate_sentiment(self, 
                          symbols: Optional[List[str]] = None,
                          hours: float = 1.0) -> Dict[str, Dict]:
        """
        Aggregate sentiment scores by symbol
        Returns dict with overall sentiment metrics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter recent sentiments
        recent_sentiments = [
            s for s in self.sentiment_history 
            if s.timestamp > cutoff_time
        ]
        
        if not symbols:
            # Get all mentioned symbols
            symbols = set()
            for s in recent_sentiments:
                symbols.update(s.symbols)
            symbols = list(symbols)
            
        results = {}
        
        for symbol in symbols:
            # Get sentiments mentioning this symbol
            symbol_sentiments = [
                s for s in recent_sentiments
                if symbol in s.symbols
            ]
            
            if symbol_sentiments:
                scores = [s.score for s in symbol_sentiments]
                confidences = [s.confidence for s in symbol_sentiments]
                
                # Weighted average by confidence
                weights = np.array(confidences)
                scores_array = np.array(scores)
                
                if weights.sum() > 0:
                    avg_sentiment = np.average(scores_array, weights=weights)
                else:
                    avg_sentiment = np.mean(scores_array)
                    
                results[symbol] = {
                    'sentiment': float(avg_sentiment),
                    'confidence': float(np.mean(confidences)),
                    'num_mentions': len(symbol_sentiments),
                    'trend': self._calculate_trend(symbol, symbol_sentiments),
                    'sources': list(set(s.source for s in symbol_sentiments))
                }
            else:
                results[symbol] = {
                    'sentiment': 0.0,
                    'confidence': 0.0,
                    'num_mentions': 0,
                    'trend': 'neutral',
                    'sources': []
                }
                
        return results
        
    def _calculate_trend(self, symbol: str, sentiments: List[SentimentScore]) -> str:
        """Calculate sentiment trend (improving/declining/stable)"""
        if len(sentiments) < 3:
            return 'neutral'
            
        # Sort by timestamp
        sorted_sentiments = sorted(sentiments, key=lambda x: x.timestamp)
        
        # Compare first half vs second half
        mid = len(sorted_sentiments) // 2
        first_half = np.mean([s.score for s in sorted_sentiments[:mid]])
        second_half = np.mean([s.score for s in sorted_sentiments[mid:]])
        
        diff = second_half - first_half
        
        if diff > 0.1:
            return 'improving'
        elif diff < -0.1:
            return 'declining'
        else:
            return 'stable'
            
    def get_market_sentiment(self) -> Dict[str, float]:
        """Get overall market sentiment"""
        recent_scores = [
            s.score for s in self.sentiment_history
            if s.timestamp > datetime.now() - timedelta(hours=1)
        ]
        
        if not recent_scores:
            return {
                'overall': 0.0,
                'confidence': 0.0,
                'sample_size': 0
            }
            
        return {
            'overall': float(np.mean(recent_scores)),
            'confidence': float(np.std(recent_scores)),  # Lower std = higher confidence
            'sample_size': len(recent_scores)
        }
        
    def adjust_strategy_weights(self, 
                              base_weights: Dict[str, float],
                              symbol_sentiments: Dict[str, Dict]) -> Dict[str, float]:
        """
        Adjust strategy weights based on sentiment
        """
        adjusted = base_weights.copy()
        
        # Calculate average sentiment across symbols
        if symbol_sentiments:
            sentiments = [s['sentiment'] for s in symbol_sentiments.values()]
            avg_sentiment = np.mean(sentiments)
            
            if avg_sentiment > 0.2:  # Bullish
                # Increase momentum/breakout strategies
                if 'MomentumBreakout' in adjusted:
                    adjusted['MomentumBreakout'] *= 1.3
                if 'VWAP_MACD_Scalper' in adjusted:
                    adjusted['VWAP_MACD_Scalper'] *= 1.2
                    
            elif avg_sentiment < -0.2:  # Bearish
                # Increase mean reversion strategies
                if 'MultiIndicatorScalper' in adjusted:
                    adjusted['MultiIndicatorScalper'] *= 1.3
                if 'Keltner_RSI_Scalper' in adjusted:
                    adjusted['Keltner_RSI_Scalper'] *= 1.2
                    
                # Reduce overall risk
                for key in adjusted:
                    adjusted[key] *= 0.8
                    
        # Normalize weights
        total = sum(adjusted.values())
        if total > 0:
            for key in adjusted:
                adjusted[key] /= total
                
        return adjusted
        
    def should_filter_signal(self, 
                           signal_direction: str,  # 'buy' or 'sell'
                           symbol: str,
                           min_sentiment_alignment: float = 0.2) -> bool:
        """
        Check if a trading signal should be filtered based on sentiment
        Returns True if signal should be BLOCKED
        """
        symbol_sentiments = self.aggregate_sentiment([symbol], hours=0.5)
        
        if symbol not in symbol_sentiments:
            return False  # No sentiment data, don't filter
            
        sentiment = symbol_sentiments[symbol]['sentiment']
        confidence = symbol_sentiments[symbol]['confidence']
        
        # Only filter if we have confident sentiment
        if confidence < 0.6:
            return False
            
        # Check alignment
        if signal_direction == 'buy' and sentiment < -min_sentiment_alignment:
            logger.warning(f"Filtering BUY signal for {symbol} due to negative sentiment: {sentiment:.2f}")
            return True
            
        if signal_direction == 'sell' and sentiment > min_sentiment_alignment:
            logger.warning(f"Filtering SELL signal for {symbol} due to positive sentiment: {sentiment:.2f}")
            return True
            
        return False
        

# Global instance
_sentiment_analyzer = None

def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get or create global sentiment analyzer"""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
    return _sentiment_analyzer
    

async def update_market_sentiment(symbols: List[str]):
    """
    Background task to continuously update sentiment
    """
    analyzer = get_sentiment_analyzer()
    
    while True:
        try:
            # Fetch latest news
            news = await analyzer.fetch_news(symbols)
            
            # Analyze sentiment
            await analyzer.analyze_news_batch(news, symbols)
            
            # Log summary
            market_sentiment = analyzer.get_market_sentiment()
            symbol_sentiments = analyzer.aggregate_sentiment(symbols)
            
            logger.info(f"Market sentiment update: Overall={market_sentiment['overall']:.2f}, "
                       f"Symbols={symbol_sentiments}")
            
            # Wait before next update
            await asyncio.sleep(300)  # 5 minutes
            
        except Exception as e:
            logger.error(f"Sentiment update error: {e}")
            await asyncio.sleep(60) 