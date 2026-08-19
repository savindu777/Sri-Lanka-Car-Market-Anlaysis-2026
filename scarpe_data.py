import re
import csv
import time
import random
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ----------------------------------------------------------------------
# CONFIG - change SEARCH_URL to scrape a different make/model/category
# ----------------------------------------------------------------------
BASE_URL = "https://riyasewana.com"
SEARCH_URL = "https://riyasewana.com/search/cars/nissan/almera"
OUTPUT_CSV = "riyasewana_nissan_almera.csv"

REQUEST_DELAY = (1.5, 3.0)   # random polite delay (seconds) between requests
MAX_RETRIES = 3
TIMEOUT = 20
ADS_PER_PAGE = 40            # riyasewana shows 40 ads per search-results page

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

FIELDS = [
    "Brand", "Model", "YOM", "Mileage", "Gear", "Fuel Type",
    "Engine (cc)", "Condition", "Ad Date", "Location", "Price", "URL",
]

# The labels exactly as they appear on a riyasewana ad detail page.
# Used both to know what to extract and as "stop" boundaries for each other,
# so extraction still works even if one field is missing from a particular ad
# (e.g. a Brand New vehicle with no Mileage).
ALL_LABELS = [
    "Location", "Year", "Mileage", "Make", "Model", "Gear",
    "Fuel Type", "Engine (cc)", "Condition", "Ad Date",
    "Options", "More Details", "views",
]


# ----------------------------------------------------------------------
# HTTP helpers
# ----------------------------------------------------------------------
def get_soup(url):
    """Fetch a URL and return a BeautifulSoup object, with retries."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as e:
            print(f"  [!] attempt {attempt}/{MAX_RETRIES} failed for {url}: {e}")
            time.sleep(2 * attempt)
    return None


def polite_sleep():
    time.sleep(random.uniform(*REQUEST_DELAY))


# ----------------------------------------------------------------------
# Search-results page parsing
# ----------------------------------------------------------------------
def get_total_pages(soup):
    """Work out how many result pages there are."""
    text = soup.get_text(" ", strip=True)

    # "Displaying 1 - 40 of 224 Search Results"
    m = re.search(r"of\s+([\d,]+)\s+Search Results", text)
    if m:
        total_ads = int(m.group(1).replace(",", ""))
        return max(1, -(-total_ads // ADS_PER_PAGE))  # ceiling division

    # Fallback: look at the pagination links for the highest page number
    page_numbers = []
    for a in soup.select("a[href*='page=']"):
        m2 = re.search(r"page=(\d+)", a.get("href", ""))
        if m2:
            page_numbers.append(int(m2.group(1)))
    return max(page_numbers) if page_numbers else 1


def get_ad_links(soup):
    """Extract all vehicle-ad detail links from a search-results page."""
    links = set()
    for a in soup.select("a[href*='/buy/']"):
        href = a.get("href")
        if not href:
            continue
        full = urljoin(BASE_URL, href)
        # keep only genuine ad pages, e.g. /buy/toyota-chr-sale-galle-12094461
        if re.search(r"/buy/[\w-]+-\d+$", full):
            links.add(full.split("?")[0])
    return links


# ----------------------------------------------------------------------
# Ad detail page parsing
# ----------------------------------------------------------------------
def extract_field(text, label):
    """
    Return the value following `label`, stopping at whichever other known
    label appears next in the text (handles ads where some fields, like
    Mileage on a brand-new vehicle, are simply absent).
    """
    other_labels = [l for l in ALL_LABELS if l != label]
    boundary = "|".join(re.escape(l) for l in other_labels)
    pattern = rf"{re.escape(label)}\s*:?\s*(.*?)\s*(?:{boundary}|$)"
    m = re.search(pattern, text)
    return m.group(1).strip(" -:") if m else ""


def scrape_ad(url):
    """Visit a single ad detail page and pull out the fields we need."""
    soup = get_soup(url)
    if soup is None:
        return None

    text = soup.get_text(" ", strip=True)

    price_match = re.search(r"Rs\.\s*[\d,]+|Negotiable", text)
    price = price_match.group(0) if price_match else ""

    return {
        "Brand": extract_field(text, "Make"),
        "Model": extract_field(text, "Model"),
        "YOM": extract_field(text, "Year"),
        "Mileage": extract_field(text, "Mileage"),
        "Gear": extract_field(text, "Gear"),
        "Fuel Type": extract_field(text, "Fuel Type"),
        "Engine (cc)": extract_field(text, "Engine (cc)"),
        "Condition": extract_field(text, "Condition"),
        "Ad Date": extract_field(text, "Ad Date"),
        "Location": extract_field(text, "Location"),
        "Price": price,
        "URL": url,
    }


# ----------------------------------------------------------------------
# Main scrape routine
# ----------------------------------------------------------------------
def scrape_all(search_url=SEARCH_URL, output_csv=OUTPUT_CSV):
    print(f"Starting scrape: {search_url}")
    first_page_soup = get_soup(search_url)
    if first_page_soup is None:
        print("Could not load the first search page. Aborting.")
        return

    total_pages = get_total_pages(first_page_soup)
    print(f"Found {total_pages} result page(s).")

    all_links = set(get_ad_links(first_page_soup))

    for page in range(2, total_pages + 1):  # scrape all pages
        page_url = f"{search_url}?page={page}"
        print(f"Fetching listing page {page}/{total_pages}: {page_url}")
        polite_sleep()
        soup = get_soup(page_url)
        if soup is None:
            continue
        links = get_ad_links(soup)
        print(f"  -> {len(links)} ad links found")
        all_links.update(links)

    print(f"\nTotal unique ads found: {len(all_links)}")

    rows = []
    for i, link in enumerate(sorted(all_links), 1):
        print(f"[{i}/{len(all_links)}] Scraping {link}")
        polite_sleep()
        data = scrape_ad(link)
        if data:
            rows.append(data)
        else:
            print(f"  [!] failed to scrape {link}")

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Saved {len(rows)} records to {output_csv}")


if __name__ == "__main__":
    scrape_all()