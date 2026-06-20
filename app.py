from flask import Flask, render_template, request, redirect, url_for, flash
import os

from ml_pipeline import get_csv_columns, run_ml_pipeline
from dl_pipeline import run_dl_pipeline
from apriori_pipeline import run_apriori_pipeline
from text_generation import generate_text
from ner_pipeline import extract_entities
from voice_qa import answer_voice_question

app = Flask(__name__)
app.secret_key = "change-this-secret-key"  # needed for flash() messages to work

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv"}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    """Check the uploaded file has a .csv extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ml", methods=["GET", "POST"])
def ml_algorithms():
    if request.method == "POST":
        file = request.files.get("dataset")
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            # Don't train yet — first let the user pick the target column
            return redirect(url_for("ml_configure", filename=file.filename))
        else:
            flash("Please upload a valid CSV file.")
            return redirect(url_for("ml_algorithms"))
    return render_template("ml.html")


@app.route("/ml/configure/<filename>")
def ml_configure(filename):
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    try:
        columns = get_csv_columns(filepath)
    except Exception as e:
        flash(f"Could not read that CSV: {e}")
        return redirect(url_for("ml_algorithms"))
    return render_template("ml_configure.html", filename=filename, columns=columns)


@app.route("/ml/train/<filename>", methods=["POST"])
def ml_train(filename):
    target_column = request.form.get("target_column")
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    try:
        results = run_ml_pipeline(filepath, target_column)
    except Exception as e:
        flash(f"Error while training models: {e}")
        return redirect(url_for("ml_configure", filename=filename))
    return render_template("ml_results.html", **results)


@app.route("/dl", methods=["GET", "POST"])
def dl_algorithms():

    if request.method == "POST":

        file = request.files.get("dataset")

        if file and allowed_file(file.filename):

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                file.filename
            )

            file.save(filepath)

            return redirect(
                url_for(
                    "dl_configure",
                    filename=file.filename
                )
            )

        else:

            flash("Please upload a valid CSV file.")

            return redirect(url_for("dl_algorithms"))

    return render_template("dl.html")


@app.route("/dl/configure/<filename>")
def dl_configure(filename):

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    try:

        columns = get_csv_columns(filepath)

    except Exception as e:

        flash(f"Could not read that CSV: {e}")

        return redirect(url_for("dl_algorithms"))

    return render_template(
        "dl_configure.html",
        filename=filename,
        columns=columns
    )

@app.route("/dl/train/<filename>", methods=["POST"])
def dl_train(filename):

    target_column = request.form.get("target_column")

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    try:

        results = run_dl_pipeline(
            filepath,
            target_column
        )

    except Exception as e:

        flash(f"Deep Learning Error: {e}")

        return redirect(
            url_for(
                "dl_configure",
                filename=filename
            )
        )

    return render_template(
        "dl_results.html",
        **results
    )


@app.route("/apriori", methods=["GET", "POST"])
def apriori_page():

    if request.method == "POST":

        file = request.files.get("dataset")

        if file and allowed_file(file.filename):

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                file.filename
            )

            file.save(filepath)

            try:

                results = run_apriori_pipeline(filepath)

            except Exception as e:

                flash(f"Apriori Error: {e}")

                return redirect(
                    url_for("apriori_page")
                )

            return render_template(
                "apriori_results.html",
                results=results
            )

        else:

            flash("Please upload a valid CSV file.")

            return redirect(
                url_for("apriori_page")
            )

    return render_template("apriori.html")



@app.route("/text-generation", methods=["GET", "POST"])
def text_generation_page():

    generated_text = None

    if request.method == "POST":

        prompt = request.form.get("prompt")

        try:

            generated_text = generate_text(prompt)

        except Exception as e:

            flash(f"Text Generation Error: {e}")

    return render_template(
        "text_generation.html",
        generated_text=generated_text
    )



@app.route("/ner", methods=["GET", "POST"])
def ner_page():

    entities = None

    if request.method == "POST":

        text = request.form.get("text")

        try:

            entities = extract_entities(text)

        except Exception as e:

            flash(f"NER Error: {e}")

    return render_template(
        "ner.html",
        entities=entities
    )

@app.route("/voice-qa", methods=["GET", "POST"])
def voice_qa_page():

    question = None
    answer = None

    if request.method == "POST":

        try:

            question, answer = answer_voice_question()

        except Exception as e:

            flash(f"Voice QA Error: {e}")

    return render_template(
        "voice_qa.html",
        question=question,
        answer=answer
    )

@app.route("/genai")
def genai_hub():
    # TODO (Phase 4-6): add sub-pages/routes for Apriori, Text Generation,
    # NER, and Voice-based Question Answering.
    return render_template("genai.html")


if __name__ == "__main__":
    app.run(debug=False)
