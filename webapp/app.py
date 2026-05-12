from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from webapp.services import (
    get_run_context,
    start_pipeline,
    submit_background,
    submit_bronze_generation,
    submit_gate1_review,
    submit_gate2_review,
    submit_gate3_review,
    submit_silver_generation,
)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "athena-local-ui")
    app.config["UPLOAD_DIR"] = str(Path(os.getcwd()) / "webapp" / "uploads")
    Path(app.config["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/upload")
    def upload_get():
        return redirect(url_for("index"))

    @app.post("/upload")
    def upload_brd():
        brd_text = request.form.get("brd_text", "").strip()
        source_databases = [db.strip() for db in request.form.get("source_databases", "").split(",") if db.strip()]
        uploaded = request.files.get("brd_file")
        input_path = None

        if uploaded and uploaded.filename:
            filename = secure_filename(uploaded.filename)
            input_path = os.path.join(app.config["UPLOAD_DIR"], filename)
            uploaded.save(input_path)

        if not brd_text and not input_path:
            flash("Provide BRD text or upload a .txt/.docx file.", "error")
            return redirect(url_for("index"))

        try:
            result = start_pipeline(
                brd_text=brd_text or None,
                input_path=input_path,
                source_databases=source_databases or None,
            )
        except Exception as exc:
            flash(f"Pipeline failed: {exc}", "error")
            return redirect(url_for("index"))

        flash(f"Run started: {result['run_id']}", "success")
        return redirect(url_for("run_detail", run_id=result["run_id"]))

    @app.get("/resume")
    def resume_get():
        return redirect(url_for("index"))

    @app.post("/resume")
    def resume_run():
        run_id = request.form.get("run_id", "").strip()
        if not run_id:
            flash("Enter a run ID to resume.", "error")
            return redirect(url_for("index"))
        return redirect(url_for("run_detail", run_id=run_id))

    @app.get("/runs/<run_id>")
    def run_detail(run_id: str):
        context = get_run_context(run_id)
        return render_template("run_detail.html", **context)

    @app.get("/runs/<run_id>/gate1")
    def gate1_get(run_id: str):
        flash("Gate 1 must be submitted from the run review form.", "error")
        return redirect(url_for("run_detail", run_id=run_id))

    @app.post("/runs/<run_id>/gate1")
    def gate1(run_id: str):
        decisions = []
        for key in request.form:
            if not key.startswith("action_"):
                continue
            item_id = key[len("action_"):]
            decisions.append(
                {
                    "item_id": item_id,
                    "action": request.form.get(f"action_{item_id}", "APPROVED"),
                    "name": request.form.get(f"name_{item_id}", ""),
                    "description": request.form.get(f"description_{item_id}", ""),
                    "reason": request.form.get(f"reason_{item_id}", ""),
                }
            )

        try:
            submit_background(run_id, "gate1", submit_gate1_review, run_id, decisions)
            flash("Gate 1 review submitted. Processing continues in the background.", "success")
        except Exception as exc:
            flash(f"Gate 1 failed: {exc}", "error")

        return redirect(url_for("run_detail", run_id=run_id))

    @app.get("/runs/<run_id>/gate2")
    def gate2_get(run_id: str):
        flash("Gate 2 must be submitted from the run review form.", "error")
        return redirect(url_for("run_detail", run_id=run_id))

    @app.post("/runs/<run_id>/gate2")
    def gate2(run_id: str):
        approved_keys = request.form.getlist("approved_table")
        if not approved_keys:
            flash("Select at least one table before submitting Gate 2.", "error")
            return redirect(url_for("run_detail", run_id=run_id))

        try:
            submit_background(run_id, "gate2", submit_gate2_review, run_id, approved_keys)
            flash("Gate 2 review submitted. Processing continues in the background.", "success")
        except Exception as exc:
            flash(f"Gate 2 failed: {exc}", "error")

        return redirect(url_for("run_detail", run_id=run_id))

    @app.get("/runs/<run_id>/gate3")
    def gate3_get(run_id: str):
        flash("Gate 3 must be submitted from the run review form.", "error")
        return redirect(url_for("run_detail", run_id=run_id))

    @app.post("/runs/<run_id>/gate3")
    def gate3(run_id: str):
        approve = request.form.get("decision", "APPROVED") == "APPROVED"
        try:
            submit_gate3_review(run_id, approve)
            flash("Gate 3 review submitted.", "success")
        except Exception as exc:
            flash(f"Gate 3 failed: {exc}", "error")

        return redirect(url_for("run_detail", run_id=run_id))

    @app.get("/runs/<run_id>/bronze")
    def bronze_get(run_id: str):
        return redirect(url_for("run_detail", run_id=run_id))

    @app.post("/runs/<run_id>/bronze")
    def bronze(run_id: str):
        try:
            submit_bronze_generation(run_id)
            flash("Bronze scripts regenerated.", "success")
        except Exception as exc:
            flash(f"Bronze generation failed: {exc}", "error")

        return redirect(url_for("run_detail", run_id=run_id))

    @app.get("/runs/<run_id>/silver")
    def silver_get(run_id: str):
        return redirect(url_for("run_detail", run_id=run_id))

    @app.post("/runs/<run_id>/silver")
    def silver(run_id: str):
        try:
            submit_silver_generation(run_id)
            flash("Silver scripts generated.", "success")
        except Exception as exc:
            flash(f"Silver generation failed: {exc}", "error")

        return redirect(url_for("run_detail", run_id=run_id))

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("ATHENA_UI_PORT", "5000")))
