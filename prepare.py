"""Turn the public Foursquare NYC check-in dataset into a Food / Quik / Shops log.

Source: Yang, Zhang, Zheng, Yu, "Modeling User Activity Preference by Leveraging User Spatial
Temporal Characteristics in LBSNs", IEEE TSMC 2014. 227K check-ins by 1,083 users in New York,
Apr 2012 - Feb 2013, with venue category and lat/lng. Mirror: huggingface.co/datasets/yyh11/FourSquare_checkin

Venue categories are mapped to Careem-like verticals:
  quik  = convenience retail you could get delivered in minutes (bodegas, delis, food & drink
          shops, pharmacies, convenience stores, markets)
  food  = restaurants, cafes, bakeries, dessert
  shops = other retail and personal services
Everything else (home, office, transit, bars, parks) is dropped. Venues with fewer than 5
check-ins are dropped so item-item models stay small. These are visits, not deliveries: a
public proxy for where people buy, not Careem data.
"""

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
URL = "https://huggingface.co/datasets/yyh11/FourSquare_checkin/resolve/main/FS_NYC.csv"
RAW = ROOT / "data" / "FS_NYC.csv"
MIN_CHECKINS = 5

QUIK = r"^(?:Food & Drink Shop|Deli / Bodega|Drugstore / Pharmacy|Convenience Store|Market)$"
FOOD = (r"Restaurant|Caf|Coffee Shop|Pizza|Burger|Diner|Bakery|Sandwich|Taco|Sushi|Dessert|Ice Cream|"
        r"Donut|Bagel|Tea Room|Steakhouse|BBQ|Snack|Breakfast|Salad|Noodle|Food Truck|Fried Chicken|Wings|"
        r"Hot Dog|Cupcake|Juice Bar|Soup|Burrito|Dumpling|Gastropub|Falafel|Brewery|^Food$|Bistro|"
        r"Frozen Yogurt|Candy|Fish & Chips")
SHOPS = r"Store|Shop|Mall|Boutique|Salon|\bSpa\b"


def load():
    """Return (items, orders). orders: user, item, vertical, hour (hours since the first check-in)."""
    if not RAW.exists():
        RAW.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL, RAW)
    d = pd.read_csv(RAW)
    cat = d["venueCategory"]
    v = pd.Series("other", index=d.index)
    v[cat.str.contains(SHOPS)] = "shops"
    v[cat.str.contains(FOOD)] = "food"
    v[cat.str.match(QUIK)] = "quik"
    d = d.assign(vertical=v)[v != "other"]
    d = d[d.groupby("venueId")["venueId"].transform("size") >= MIN_CHECKINS]

    t = pd.to_datetime(d["utcTimestamp"], format="%a %b %d %H:%M:%S +0000 %Y")
    d = d.assign(hour=((t - t.min()).dt.total_seconds() / 3600).round(3))
    d = d.sort_values(["vertical", "venueId"])
    items = (d.groupby("venueId", sort=False)
             .agg(vertical=("vertical", "first"), category=("venueCategory", "first"),
                  lat=("latitude", "median"), lng=("longitude", "median"))
             .reset_index())
    items["item"] = np.arange(len(items))
    d["user"] = pd.factorize(d["userId"], sort=True)[0]
    d["item"] = d["venueId"].map(items.set_index("venueId")["item"])
    orders = d[["user", "item", "vertical", "hour"]].sort_values(["user", "hour", "item"]).reset_index(drop=True)
    return items, orders


if __name__ == "__main__":
    items, orders = load()
    print(items["vertical"].value_counts().to_string())
    print(orders["vertical"].value_counts().to_string())
    print(f"{len(orders)} check-ins, {orders['user'].nunique()} users, {len(items)} venues")
