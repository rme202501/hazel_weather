import pytesseract
import cv2

IMAGE_PATH = "data/12-27/12-27weather1.png"

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

text = pytesseract.image_to_string(gray)
print("----- RAW OCR OUTPUT -----")
lines = [line.strip() for line in text.splitlines() if line.strip()]
cleaned = "\n".join(lines)
print(cleaned)