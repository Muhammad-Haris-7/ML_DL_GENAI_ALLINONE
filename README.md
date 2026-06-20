# AI Project Hub — Phase 1 Skeleton

## What this is
A Flask app skeleton with three sections (Machine Learning, Deep Learning,
Generative AI). Right now the ML and DL pages accept a CSV upload and save
it — model training logic will be added in later phases.

## How to run it

1. Open a terminal in this folder.
2. (Recommended) create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the app:
   ```
   python app.py
   ```
5. Open your browser to: http://127.0.0.1:5000

## What to check
- Clicking all three navbar links (Machine Learning / Deep Learning /
  Generative AI) should work without errors.
- Uploading a CSV on the ML or DL page should show a success message and
  save the file into the `uploads/` folder.

## Next phases
- Phase 2: Add real ML model training to `/ml`
- Phase 3: Add neural network training to `/dl`
- Phase 4: Add Apriori (association rule mining)
- Phase 5: Add Text Generation + NER using Hugging Face pipelines
- Phase 6: Add voice-based Question Answering
- Phase 7: Polish and deploy to Render
