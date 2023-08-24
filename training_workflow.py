import mlrun


def pipeline(
    input_path,
    model_transcribe,
    pii_model,
    pii_entities,
    qa_model,
    qa_questions,
    qa_questions_columns,
):
    project = mlrun.get_current_project()
    transcription_run = project.run_function(
        function="transcribe",
        handler="transcribe",
        params={
            "input_path": "/v3io/bigdata/sample-data",
            "decoding_options": {"fp16": False},
            "model_name": "tiny",
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
            "model": "whole",
            "output_path": "./cleaned_data",
            "output_suffix": "output",
            "html_key": "highlighted",
            "entities": ["PERSON", "EMAIL", "PHONE", "LOCATION", "ORGANIZATION"],
            "score_threshold": 0.5,
        },
        returns=["output_path: path", "rpt_json: file", "errors: file"],
    )

    question_answering_run = project.run_function(
        function="question-answering",
        handler="answer_questions",
        inputs={"input_path": transcription_run.outputs["transcriptions"]},
        params={
            "model": "tiiuae/falcon-7b-instruct",
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
            "questions": [
                "Classify the Customer issue in 1-4 words.",
                "Write a 50-100 word summary of the text.",
                "Was the Customer issue fixed, Yes or No?",
                "In one word, was the Customer tone Positive, Negative or Natural?",
                "In one word, was the Call Center Agent tone Positive, Negative or Natural?",
            ],
            "questions_columns": [
                "Issue",
                "Summary",
                "is_fixed",
                "customer_tone",
                "agent_tone",
            ],
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
