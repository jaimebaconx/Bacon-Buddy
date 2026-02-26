import internetarchive as ia
import os

# ── SETTINGS ──
DOWNLOAD_PATH = r"E:\BaconBuddy\campfire\audio"
COLLECTION = "oldtimeradio"
SEARCH_TERM = "inner sanctum mysteries"
MAX_EPISODES = 3

os.makedirs(DOWNLOAD_PATH, exist_ok=True)

print(f"Searching for: {SEARCH_TERM}")
results = ia.search_items(f"collection:{COLLECTION} AND {SEARCH_TERM}")

count = 0
for result in results:
    if count >= MAX_EPISODES:
        break
    
    identifier = result['identifier']
    print(f"Downloading: {identifier}")
    
    item = ia.get_item(identifier)
    
    for file in item.files:
        # Grab VBR MP3s only, skip ogg, png, xml, torrent
        if file['name'].endswith('.mp3') and 'kf' not in file['name'] and '64kb' not in file['name']:
            print(f"  Downloading: {file['name']}")
            item.download(
                files=[file['name']],
                destdir=DOWNLOAD_PATH,
                no_directory=True
            )
    
    count += 1

print(f"Done!")