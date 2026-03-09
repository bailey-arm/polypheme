import requests
import time
import csv
from datetime import datetime
import json
import pandas as pd
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeopoliticalMarketScraper:
    def __init__(self, markets: List[Dict]):
        """
        Initialize scraper with list of markets
        
        markets: List of dicts with keys:
            - name: Market name/slug
            - token_id: CLOB token ID for the outcome to track
            - question: The market question
            - outcome: Which outcome to track (e.g., "Yes" or "No")
        """
        self.markets = markets
        self.data = {market['token_id']: [] for market in markets}
        
    def fetch_price(self, token_id: str) -> Dict:
        """Fetch current price for a token from CLOB API"""
        url = f"https://clob.polymarket.com/price"
        
        try:
            # Get bid price (probability of outcome)
            response = requests.get(
                url, 
                params={"token_id": token_id, "side": "buy"},
                timeout=10
            )
            
            if response.status_code == 200:
                bid_price = response.json().get('price')
                
                # Get ask price
                response = requests.get(
                    url,
                    params={"token_id": token_id, "side": "sell"},
                    timeout=10
                )
                
                ask_price = response.json().get('price') if response.status_code == 200 else None
                
                # Get order book for more detail
                book_response = requests.get(
                    f"https://clob.polymarket.com/book",
                    params={"token_id": token_id},
                    timeout=10
                )
                
                book_data = book_response.json() if book_response.status_code == 200 else {}
                
                return {
                    'timestamp': datetime.now().isoformat(),
                    'token_id': token_id,
                    'bid_price': float(bid_price) if bid_price else None,
                    'ask_price': float(ask_price) if ask_price else None,
                    'mid_price': (float(bid_price) + float(ask_price)) / 2 if bid_price and ask_price else None,
                    'bid_size': book_data.get('bids', [{}])[0].get('size') if book_data.get('bids') else None,
                    'ask_size': book_data.get('asks', [{}])[0].get('size') if book_data.get('asks') else None,
                }
            else:
                logger.error(f"Error fetching price for {token_id}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Exception fetching {token_id}: {e}")
            return None
    
    def fetch_midpoint(self, token_id: str) -> Dict:
        """Alternative method using midpoint endpoint"""
        url = f"https://clob.polymarket.com/midpoint"
        
        try:
            response = requests.get(url, params={"token_id": token_id}, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'timestamp': datetime.now().isoformat(),
                    'token_id': token_id,
                    'midpoint': float(data.get('midpoint')) if data.get('midpoint') else None
                }
            else:
                logger.error(f"Error fetching midpoint for {token_id}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Exception fetching midpoint {token_id}: {e}")
            return None
    
    def run_minute_level(self, duration_minutes: int = 60):
        """
        Run scraper at minute-level intervals
        
        Args:
            duration_minutes: How many minutes to collect data for
        """
        logger.info(f"Starting minute-level data collection for {len(self.markets)} markets")
        logger.info(f"Will collect for {duration_minutes} minutes")
        
        start_time = time.time()
        iterations = 0
        
        while iterations < duration_minutes:
            current_iteration = iterations + 1
            logger.info(f"Iteration {current_iteration}/{duration_minutes}")
            
            # Fetch data for each market
            for market in self.markets:
                token_id = market['token_id']
                
                # Try primary price endpoint
                price_data = self.fetch_price(token_id)
                
                if price_data and price_data.get('mid_price'):
                    # Add market metadata
                    price_data['market_name'] = market['name']
                    price_data['question'] = market['question']
                    price_data['outcome'] = market['outcome']
                    
                    self.data[token_id].append(price_data)
                    logger.info(f"  {market['name']}: {price_data['mid_price']:.3f} "
                               f"(bid: {price_data['bid_price']:.3f}, ask: {price_data['ask_price']:.3f})")
                else:
                    # Fallback to midpoint endpoint
                    midpoint_data = self.fetch_midpoint(token_id)
                    if midpoint_data:
                        midpoint_data['market_name'] = market['name']
                        midpoint_data['question'] = market['question']
                        midpoint_data['outcome'] = market['outcome']
                        # Add price fields for consistency
                        midpoint_data['bid_price'] = None
                        midpoint_data['ask_price'] = None
                        midpoint_data['mid_price'] = midpoint_data.get('midpoint')
                        midpoint_data['bid_size'] = None
                        midpoint_data['ask_size'] = None
                        
                        self.data[token_id].append(midpoint_data)
                        logger.info(f"  {market['name']}: {midpoint_data['midpoint']:.3f} (via midpoint)")
                    else:
                        logger.error(f"  Failed to get data for {market['name']}")
            
            # Wait exactly 60 seconds for next iteration
            if iterations < duration_minutes - 1:
                time.sleep(60 - (time.time() - start_time) % 60)
            
            iterations += 1
        
        logger.info(f"Collection complete. Collected {sum(len(d) for d in self.data.values())} data points")
        return self.data
    
    def save_to_csv(self, filename_prefix: str = "geopolitical_odds"):
        """Save collected data to CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for token_id, records in self.data.items():
            if not records:
                continue
                
            # Find market name for this token
            market = next((m for m in self.markets if m['token_id'] == token_id), None)
            if not market:
                continue
            
            # Create filename
            safe_name = market['name'].replace(' ', '_').replace('/', '_')[:30]
            filename = f"{filename_prefix}_{safe_name}_{timestamp}.csv"
            
            # Convert to DataFrame and save
            df = pd.DataFrame(records)
            
            # Reorder columns for readability
            cols = ['timestamp', 'market_name', 'question', 'outcome', 
                    'mid_price', 'bid_price', 'ask_price', 'bid_size', 'ask_size']
            available_cols = [c for c in cols if c in df.columns]
            df = df[available_cols]
            
            df.to_csv(filename, index=False)
            logger.info(f"Saved {len(df)} records to {filename}")
            
            # Also save metadata summary
            summary = {
                'market_name': market['name'],
                'question': market['question'],
                'outcome': market['outcome'],
                'token_id': token_id,
                'first_record': records[0]['timestamp'],
                'last_record': records[-1]['timestamp'],
                'record_count': len(records),
                'avg_price': sum(r['mid_price'] for r in records if r['mid_price']) / len([r for r in records if r['mid_price']]),
                'min_price': min(r['mid_price'] for r in records if r['mid_price']),
                'max_price': max(r['mid_price'] for r in records if r['mid_price'])
            }
            
            summary_filename = f"{filename_prefix}_{safe_name}_summary_{timestamp}.json"
            with open(summary_filename, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Saved summary to {summary_filename}")

# Example usage - find 5 geopolitical markets to track
def find_top_geopolitical_markets():
    """Find top 5 geopolitical markets by volume"""
    
    # First, use the Gamma API to find active geopolitical markets
    url = "https://gamma-api.polymarket.com/markets"
    
    # Common geopolitical keywords
    keywords = ["russia", "ukraine", "israel", "iran", "china", "taiwan", 
                "election", "war", "conflict", "ceasefire"]
    
    all_markets = []
    
    for keyword in keywords[:3]:  # Try a few keywords
        params = {
            "limit": 10,
            "title": keyword,
            "closed": False,
            "order": "volume",
            "ascending": False
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            markets = response.json()
            
            for market in markets:
                token_ids = market.get('clobTokenIds', [])
                if isinstance(token_ids, str):
                    token_ids = json.loads(token_ids)
                if len(token_ids) >= 2:
                    # For yes/no markets, track the "Yes" outcome (first token)
                    all_markets.append({
                        'name': market['slug'],
                        'token_id': token_ids[0],  # Yes token
                        'question': market['question'],
                        'outcome': 'Yes',
                        'volume': float(market.get('volume', 0)),
                        'endDate': market.get('endDate')
                    })
                    
                    # Also optionally track the "No" outcome
                    if len(token_ids) > 1:
                        all_markets.append({
                            'name': f"{market['slug']}_no",
                            'token_id': token_ids[1],  # No token
                            'question': market['question'],
                            'outcome': 'No',
                            'volume': float(market.get('volume', 0)),
                            'endDate': market.get('endDate')
                        })
    
    # Sort by volume and take top 5 unique markets
    all_markets.sort(key=lambda x: x['volume'], reverse=True)
    
    # Deduplicate by question and outcome
    unique_markets = []
    seen = set()
    
    for market in all_markets:
        key = f"{market['question']}_{market['outcome']}"
        if key not in seen:
            seen.add(key)
            unique_markets.append(market)
            
        if len(unique_markets) >= 5:
            break
    
    return unique_markets[:5]

# Main execution
if __name__ == "__main__":
    # Find top geopolitical markets
    print("Finding top geopolitical markets...")
    markets_to_track = find_top_geopolitical_markets()
    
    print(f"Found {len(markets_to_track)} markets to track:")
    for i, market in enumerate(markets_to_track, 1):
        print(f"{i}. {market['question']} - {market['outcome']} "
              f"(Volume: ${market['volume']/1e6:.1f}M)")
    
    # Initialize scraper
    scraper = GeopoliticalMarketScraper(markets_to_track)
    
    # Run for 60 minutes (adjust as needed)
    data = scraper.run_minute_level(duration_minutes=60)
    
    # Save results
    scraper.save_to_csv("geopolitical_odds")