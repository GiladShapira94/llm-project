import mlrun


def pipeline(
    input_path,
    transcribe_model,
    pii_model,
    pii_entities,
    qa_model,
    qa_questions,
    qa_questions_columns,
):
    project = mlrun.get_current_project()
    transcribe_func = project.get_function("transcribe")
    transcribe_func.apply(mlrun.auto_mount())
    transcription_run = project.run_function(
        function="transcribe",
        handler="transcribe",
        params={
            "input_path": input_path,
            "decoding_options": {"fp16": False},
            "model_name": transcribe_model,
            "output_directory": "./transcripted_data",
        },
        returns=[
            "transcriptions: path",
            "transcriptions_df: dataset",
            {"key": "transcriptions_errors", "artifact_type": "file"},
        ],
    )

    pii_recognizing_run = project.run_function(
        function="pii-recognizer",
        handler="recognize_pii",
        inputs={"input_path": transcription_run.outputs["transcriptions"]},
        params={
            "model": pii_model,
            "output_path": "./cleaned_data",
            "output_suffix": "output",
            "html_key": "highlighted",
            "entities": pii_entities,
            "score_threshold": 0.5,
        },
        returns=["output_path: path", "rpt_json: file", "errors: file"],
    )

    question_answering_run = project.run_function(
        function="question-answering",
        handler="answer_questions",
        inputs={"input_path": pii_recognizing_run.outputs["output_path"]},
        params={
            "model": qa_model,
            "model_kwargs": {
                "device_map": "auto",
                "load_in_8bit": True,
            },
            "text_wrapper": (
                "Given the following conversation between a Customer and a Call Center Agent:\n"
                "-----\n"
                "{}\n"
                "-----"
            ),
            "questions": qa_questions,
            "questions_columns": qa_questions_columns,
            "generation_config": {
                "do_sample": True,
                "temperature": 0.8,
                "top_p": 0.9,
                "early_stopping": True,
                "max_new_tokens": 150,
            },
        },
        returns=[
            "question_answering_df: dataset",
            "question_answering_errors: result",
        ],
    )
    postprocess_run = project.run_function(
        function="postprocess",
        handler="postprocess",
        inputs={
            "transcript_dataset": transcription_run.outputs["transcriptions_df"],
            "qa_dataset": question_answering_run.outputs["question_answering_df"],
        },
        returns=["final_df: dataset"],
    )
