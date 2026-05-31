import json
import re
import feedparser
import importlib
word_vectoring = importlib.import_module("Word vectoring pro")

NICHE_BLOGS = {
    "Technology": [
        "https://techcrunch.com/feed/",
        "https://www.theverge.com/rss/index.xml",
        "https://feeds.feedburner.com/arstechnica/index",
        "https://www.wired.com/feed/rss",
        "https://www.technologyreview.com/feed/"
    ],
    "Automotive": [
        "https://www.autoblog.com/rss.xml",
        "https://www.caranddriver.com/rss/all.xml",
        "https://www.motortrend.com/feed/",
        "https://jalopnik.com/rss"
    ],
    "Culinary": [
        "https://www.seriouseats.com/feeds/search?feed=1",
        "https://www.pinchofyum.com/feed",
        "https://www.thekitchn.com/main.rss",
        "https://www.eater.com/rss/index.xml"
    ],
    "Finance_Business": [
        "https://feeds.feedburner.com/harvardbusiness",
        "https://blogs.forbes.com/business/feed/",
        "https://www.investopedia.com/rss/"
    ],
    "Health_Wellness": [
        "https://rss.healthline.com/healthline/most-popular",
        "https://www.mindbodygreen.com/rss/feed.xml",
        "https://rssfeeds.webmd.com/rss/rss.aspx?source=1",
        "https://feeds.feedburner.com/zenhabits"
    ]
}

def clean_html(raw_html):
    """Strips HTML tags so text doesn't corrupt vector alignments"""
    clean_re = re.compile('<.*?>|&([a-z0-9]+|#[0-9]+|#x[0-9a-f]+);')
    text = re.sub(clean_re, '', raw_html)
    return re.sub(r'\s+', ' ', text).strip()

def run_automated_training():
    print("[System] Booting up training engine from Word vectoring pro...")
    word_vectors, known_words = word_vectoring.load_brain_memory()
    
    total_articles = 0
    for niche, feeds in NICHE_BLOGS.items():
        print(f"\n [Connecting] Ingesting Niche: {niche.upper()}")
        for feed_url in feeds:
            print(f" Pulling latest entries from: {feed_url}")
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    
                    full_text = title + " " + summary
                    if 'content' in entry:
                        full_text += " " + entry.content[0].value
                    clean_text = clean_html(full_text)
                    if clean_text:
                        word_vectors, known_words = word_vectoring.gobbler(clean_text, word_vectors, known_words)
                        total_articles += 1
            except Exception as e:
                print(f" Connection failed for {feed_url}: {e}")
    print(f"\n[✓] Finished reading {total_articles} articles across niches.")
    print("\nShutting down training loops... Initiating synchronization pipeline.")
    word_vectoring.save_brain_memory(word_vectors, known_words)
    word_vectors, known_words = word_vectoring.review_stop_words(word_vectors, known_words)
    word_vectoring.save_brain_memory(word_vectors, known_words)
    
    print("\n[*] Processing vector alignments via round-robin cluster calculations...")
    final_relations = word_vectoring.grouper(word_vectors)
    
    with open(word_vectoring.HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_relations, f, indent=4) 
        
    print(f"\n[✓] Done! Files 'semantic_brain.json' and 'relations_history.json' updated.")

if __name__ == "__main__":
    run_automated_training()
