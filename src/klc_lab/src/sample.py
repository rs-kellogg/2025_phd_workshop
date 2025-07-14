import sys
import cowsay

if len(sys.argv) != 2:
    print("Usage: python helllo.py [name]")
    sys.exit(1)

name = sys.argv[1]
cowsay.cow(f"helllo {name}! Great work today!")

# sleep for 1 minute
import time
time.sleep(60)

cowsay.cow(f"Wake up now!")

