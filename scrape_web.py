import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("FIRECRAWL_API_KEY")
API_URL = "https://api.firecrawl.dev/v1/scrape"

def scrape_public_pages():
    urls = [
        "https://www.daraz.pk/help/shipping-policy",
        "https://www.amazon.com/gp/help/customer/display.html",
        "https://www.apple.com/legal/privacy/"
    ]
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    for url in urls:
        try:
            print(f"🔄 Scraping: {url}")
            
            payload = {
                "url": url,
                "formats": ["markdown"]
            }
            
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                # 🔥 Firecrawl ke response mein data 'data' key mein aata hai
                content = data.get("data", {}).get("markdown", "")
                
                # Agar upar se na mile toh alternate structure try karo
                if not content:
                    content = data.get("content", "")
                
                if content:
                    # Safe filename banao
                    safe_name = url.split('/')[-1].replace('?', '_').replace('=', '_')
                    if not safe_name:
                        safe_name = "page"
                    filename = f"documents/scraped_{safe_name}.md"
                    
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"✅ Saved: {filename}")
                else:
                    print(f"⚠️ No markdown content found for {url}")
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                
            time.sleep(2)  # Rate limit ke liye
            
        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")

if __name__ == "__main__":
    # Ensure documents folder exists
    if not os.path.exists("documents"):
        os.makedirs("documents")
    scrape_public_pages()