import requests
from pathlib import Path
import csv

book_list = []
with open("data/books_list.csv", "r", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row:  # Ensure the row is not empty
            book_list.append(row[1])
print(book_list)

# book_folder = Path("./data/books")
# book_folder.mkdir(parents=True, exist_ok=True)

for i, url in enumerate(book_list):
    response = requests.get(url)
    if response.status_code == 200:
        filename = url.split("/")[-2] + ".html"
        filepath = book_folder / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"Book {i+1} saved successfully.")
    else:
        print(f"Failed to retrieve book {i+1}. Status code: {response.status_code}")
