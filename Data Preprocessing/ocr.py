import pytesseract
import shutil
import os
import sys
import random
import subprocess
from tqdm import tqdm
try:
  from PIL import Image
except ImportError:
  import Image

ROOT_FOLDER = './all_bail_txt/' #target txt folder
SOURCE_FOLDER = './all_pdf_download/' #source folder of all pdfs
def run(param):
    district = param[0]
    subcourt = param[1]
    TARGET_FOLDER = f'{ROOT_FOLDER}{district.lower().replace(" ","_")}/{subcourt.lower().replace(" ","_").replace(",","")}/'
    print(TARGET_FOLDER)

    temp_file = district.lower().replace(" ","_") +  '.tiff'
    try:
        os.makedirs(TARGET_FOLDER)
    except:
        pass
    source_path = f'{SOURCE_FOLDER}{district.lower().replace(" ","_")}/{subcourt.lower().replace(" ","_").replace(",","")}/'

    for filename in tqdm(os.listdir(source_path)):
        if filename.endswith(".pdf"):
            try:               
                file_path = os.path.join(source_path, filename)
                if os.path.exists( TARGET_FOLDER + str(filename.split(".")[0]) + ".txt"):
                    continue
                subprocess.run(['gs', '-dNOPAUSE', '-q', '-r300x300', '-sDEVICE=tiffg4', '-dBATCH', str('-sOutputFile='+temp_file), file_path])
                subprocess.run(['tesseract', temp_file , os.path.join(TARGET_FOLDER + str(filename.split(".")[0])), '-l', 'Devanagari'])
                
            except Exception as e:
                print(e, filename)
                continue

def main(idx):
    import json
    all_values = []
    with open("districts.json") as f:
        districts = json.load(f)
    print("Districs are:", districts)
    for district in districts:
        with open("subcourts.json") as f:
            subcourts = json.load(f)
        # get data for each subcourt and district combo...
        for subcourt in subcourts[district]:
            print("Doing:", district, subcourt)
            all_values.append([district, subcourt])
    chunks = [ all_values[0:20] , all_values[20:40] , all_values[40:60] , all_values[60:80] , all_values[80:100], all_values[100:120], all_values[120:140], all_values[140:]]

    for vals in chunks[int(idx)]:
        run(vals)
        
if __name__ == "__main__":
    main(sys.argv[1])