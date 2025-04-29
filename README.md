#Project Files
main.py: Contains the main code that handles image uploads, runs OCR (Optical Character Recognition), and returns the text.

Working_Main.ipynb: A Jupyter Notebook that shows how the app works step by step.

results/: Folder where the results of the OCR (text and JSON files) are saved.

#How It Works (Simple Flow)
User uploads an image + name

FastAPI processes the image using Pytesseract to extract text.

The extracted text is sent back to the user.

