import urllib.request
from html.parser import HTMLParser
import json
import concurrent.futures

class LinksParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = {}
        self.current_href = None
        
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            d = dict(attrs)
            href = d.get('href', '')
            if 'national-team' in href or '-federation' in href:
                self.current_href = href
                
    def handle_data(self, data):
        if self.current_href:
            team_name = data.strip()
            if team_name and team_name not in ['Main', 'white', 'unofficial']:
                self.links[team_name] = self.current_href
                self.current_href = None

class ImgParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.img = None
    def handle_starttag(self, tag, attrs):
        if tag == 'img' and not self.img:
            d = dict(attrs)
            src = d.get('src', '')
            if ('national' in src or 'logo' in src or 'federation' in src) and src.endswith('.png'):
                self.img = src if src.startswith('http') else 'https://football-logos.cc' + src

def fetch_html(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        return urllib.request.urlopen(req, timeout=10).read().decode('utf-8')
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

def get_logo(team, url):
    html = fetch_html(url)
    parser = ImgParser()
    parser.feed(html)
    return team, parser.img

print("Fetching main page...")
main_html = fetch_html("https://football-logos.cc/tournaments/fifa-world-cup-2026/unofficial/")
p = LinksParser()
p.feed(main_html)

logos = {}
print(f"Found {len(p.links)} teams, fetching logos...")
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(get_logo, team, url) for team, url in p.links.items()]
    for future in concurrent.futures.as_completed(futures):
        team, img = future.result()
        if img:
            logos[team] = img

with open("data/logos.json", "w") as f:
    json.dump(logos, f, indent=2)

print("Saved logos to data/logos.json")
