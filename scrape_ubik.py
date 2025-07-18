import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Base website URL
base_url = "https://www.ubiksolution.com"

# List of known subpages to scrape
subpages = ["/", "/about", "/products", "/services", "/solutions", "/contact", "/team"]

# Dictionary to store scraped content
scraped_data = {}

# Scraping Function
def scrape_page(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        # Extract only visible text
        return soup.get_text(separator='\n', strip=True)
    except Exception as e:
        return f"Error: {str(e)}"

# Loop through each page
for path in subpages:
    full_url = urljoin(base_url, path)
    print(f"Scraping {full_url}")
    scraped_data[path] = scrape_page(full_url)

# Save scraped data to a file
with open("ubik_scraped_content.txt", "w", encoding="utf-8") as f:
    for path, content in scraped_data.items():
        f.write(f"--- {path} ---\n{content}\n\n")
